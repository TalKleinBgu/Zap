"""
Zap Product Deduplication — Pipeline v3
========================================
Stage 0: LLM-based Hebrew→English name normalization  (cached, one-time per name)
Stage 1: Multilingual Embeddings + FAISS ANN          (normalized names → embeddings)
Stage 2: Category-aware Union-Find clustering
Stage 3: LLM cluster refinement + canonical naming    (normalized names, no Hebrew)
Stage 4: Merge & lowest price per group

Key design decisions:
- Normalized names are used consistently: for embedding (stage 1) AND LLM refinement (stage 3).
  Original names are preserved only for display in the output JSON/CSV.
- Two caches: normalizations.json and embeddings.json — re-runs make zero API calls.
- FAISS IndexFlatIP on L2-normalized vectors gives cosine similarity via inner product.
"""

import csv
import json
import os
import numpy as np
import faiss
import matplotlib.pyplot as plt
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD        = 0.72
TOP_K                       = 10
INPUT_CSV                   = "data/products_benchmark.csv"
OUTPUT_CSV                  = "output/deduplicated_products.csv"
EMBEDDING_CACHE_PATH        = "cache/embeddings.json"
NORMALIZATION_CACHE_PATH    = "cache/normalizations.json"
EMBEDDING_MODEL             = "text-embedding-3-small"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── Hebrew → English normalization via LLM ────────────────────────────────────
# Rather than a hardcoded brand dict, we ask the LLM to translate any
# Hebrew-containing product name to its standard English form before embedding.
# Results are cached in cache/normalizations.json so each name is only
# translated once across all runs.

_normalization_cache: dict[str, str] = {}

HEBREW_RANGE = range(0x0590, 0x05FF + 1)

def _contains_hebrew(text: str) -> bool:
    return any(ord(c) in HEBREW_RANGE for c in text)


def load_normalization_cache() -> None:
    global _normalization_cache
    if os.path.exists(NORMALIZATION_CACHE_PATH):
        with open(NORMALIZATION_CACHE_PATH, encoding="utf-8") as f:
            _normalization_cache = json.load(f)
        print(f"   Loaded {len(_normalization_cache)} cached normalizations")


def save_normalization_cache() -> None:
    os.makedirs(os.path.dirname(NORMALIZATION_CACHE_PATH), exist_ok=True)
    with open(NORMALIZATION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_normalization_cache, f, ensure_ascii=False, indent=2)


def normalize_names(names: list[str]) -> list[str]:
    """
    Translate Hebrew-containing product names to standard English before embedding.
    Pure-English names pass through unchanged without touching the API.
    Results are cached — only uncached Hebrew names hit the API.

    Note: this only affects what gets embedded. Original names are used everywhere else.
    """
    missing = [n for n in names if _contains_hebrew(n) and n not in _normalization_cache]

    if missing:
        payload = [{"index": i, "name": n} for i, n in enumerate(missing)]
        prompt = (
            "You are a product name normalizer for an Israeli price-comparison site.\n"
            "Translate each Hebrew or mixed Hebrew/English product name to its standard "
            "English product name. Rules:\n"
            "  • Keep all model numbers, specs, and capacities exactly (256GB, 65\", 9KG).\n"
            "  • Translate the brand and product name to English "
            "(e.g. 'לוג'יטק G Pro X Superlight 2' → 'Logitech G Pro X Superlight 2', "
            "'בוז קוויאט קומפורט 45' → 'Bose Quiet Comfort 45').\n"
            "  • Do NOT invent specs that are not in the original name.\n\n"
            f"Names:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"results": [{"index": 0, "normalized": "..."}, ...]}'
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        for item in data["results"]:
            original = missing[item["index"]]
            _normalization_cache[original] = item["normalized"]

    return [_normalization_cache.get(n, n) for n in names]


# ── Embedding cache ───────────────────────────────────────────────────────────
_embedding_cache: dict[str, list[float]] = {}


def load_embedding_cache() -> None:
    global _embedding_cache
    if os.path.exists(EMBEDDING_CACHE_PATH):
        with open(EMBEDDING_CACHE_PATH, encoding="utf-8") as f:
            _embedding_cache = json.load(f)
        print(f"   Loaded {len(_embedding_cache)} cached embeddings")


def save_embedding_cache() -> None:
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_embedding_cache, f, ensure_ascii=False)


# ── Stage 1: Embeddings ───────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Batch embed texts with cache.
    Only calls the API for texts not already cached.
    Returns L2-normalized (n, dim) float32 array.
    """
    missing = [t for t in texts if t not in _embedding_cache]
    if missing:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=missing)
        for text, item in zip(missing, response.data):
            _embedding_cache[text] = item.embedding

    vecs = np.array([_embedding_cache[t] for t in texts], dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


# ── Stage 2: FAISS ANN + Union-Find clustering ────────────────────────────────

def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    parent[_find(parent, x)] = _find(parent, y)


def cluster_by_category(products: list[dict]) -> list[list[dict]]:
    """
    Groups products into clusters using per-category FAISS ANN.
    LLM-normalizes Hebrew names to English before embedding (cached per name).
    Only products in the same category are compared.
    Returns flat list of clusters (each cluster is a list of products).
    """
    by_category: dict[str, list[dict]] = {}
    for p in products:
        by_category.setdefault(p["category"], []).append(p)

    all_clusters: list[list[dict]] = []

    for category, items in by_category.items():
        n = len(items)
        # Normalize Hebrew → English; store on each product so stage 3 reuses it
        norm_names = normalize_names([p["name"] for p in items])
        for p, norm in zip(items, norm_names):
            p["_norm"] = norm

        cached = sum(1 for name in norm_names if name in _embedding_cache)
        api_calls = n - cached
        suffix = f" ({cached} cached, {api_calls} new)" if cached > 0 else ""
        print(f"   [{category}] {n} products → embedding{suffix}...", flush=True)

        vecs = get_embeddings(norm_names)
        parent = list(range(n))

        if n > 1:
            k = min(TOP_K + 1, n)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            scores_matrix, indices_matrix = index.search(vecs, k)

            for i in range(n):
                for rank in range(1, k):
                    j = int(indices_matrix[i][rank])
                    score = float(scores_matrix[i][rank])
                    if score >= SIMILARITY_THRESHOLD:
                        _union(parent, i, j)

        clusters: dict[int, list[dict]] = {}
        for i, p in enumerate(items):
            clusters.setdefault(_find(parent, i), []).append(p)

        all_clusters.extend(clusters.values())

    return all_clusters


# ── Stage 3: LLM cluster refinement ──────────────────────────────────────────

def llm_refine(cluster: list[dict]) -> list[dict]:
    """
    Send the full cluster to GPT-4o-mini.
    Returns list of {canonical_name, products} — may split the cluster on false positives.

    Uses normalized (English) names so the prompt needs no Hebrew-handling instructions.
    Original names are preserved in the products for output.
    """
    # Use normalized names for the LLM — Hebrew was already translated in stage 0
    items_payload = [{"id": p["id"], "name": p.get("_norm", p["name"])} for p in cluster]
    id_to_product = {str(p["id"]): p for p in cluster}

    prompt = f"""You are a product deduplication expert for Zap, an Israeli price-comparison site.

These products were flagged as potentially identical. Your tasks:
1. Group them by truly identical product (same brand + model + storage/capacity/screen-size).
   Model number formatting variants may refer to the same product — check carefully
   (e.g. "ODYSSEY-G7 32inch 240Hz" and "32G75T Odyssey G7" are the same monitor).
   - Storage size (e.g. 256GB vs 512GB) IS a different product — keep separate.
   - Screen size (e.g. 55" vs 65") IS a different product — keep separate.
   - Capacity (e.g. 9KG vs 11KG, 559L vs 635L) IS a different product — keep separate.
   - Color variants are the same product — merge them.
   - Different model numbers (e.g. S22 vs S23, XM4 vs XM5) ARE different products.
2. For each group, provide a clean canonical English name including the key spec
   (e.g. "Samsung Galaxy S24 Ultra 256GB", "LG OLED C4 65 inch").

Products:
{json.dumps(items_payload, ensure_ascii=False, indent=2)}

Respond ONLY with valid JSON, no extra text:
{{
  "groups": [
    {{"canonical_name": "...", "ids": ["1", "2", "3"]}}
  ]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    result = []
    seen_ids: set[str] = set()
    for group in data["groups"]:
        members = [id_to_product[str(i)] for i in group["ids"] if str(i) in id_to_product]
        if members:
            result.append({"canonical_name": group["canonical_name"], "products": members})
            seen_ids.update(str(p["id"]) for p in members)

    # Safety: if the LLM dropped any product IDs, add them back as singletons
    for pid, p in id_to_product.items():
        if pid not in seen_ids:
            result.append({"canonical_name": p.get("_norm", p["name"]), "products": [p]})

    return result


# ── Stage 4: Min price & best deal ───────────────────────────────────────────

def build_record(canonical_name: str, products: list[dict]) -> dict:
    sources = sorted(
        [{"id": p["id"], "name": p["name"], "price": float(p["price"])}
         for p in products],
        key=lambda x: x["price"]
    )
    best = sources[0]
    return {
        "canonical_name":  canonical_name,
        "min_price":        best["price"],
        "best_deal_id":     best["id"],
        "best_deal_name":   best["name"],
        "category":         products[0]["category"],
        "duplicate_count":  len(products),
        "sources":          sources,
    }


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(output: list[dict], path: str, products: list[dict] | None = None) -> None:
    """
    Write a flat CSV — one row per deduplicated product.
    Variants are pipe-separated in the last column.
    When products (with group_id) are provided, a gt_min_price column is added.
    """
    # Build GT min-price lookup: group_id -> min price across all true members
    id_to_group: dict[str, str] = {}
    group_to_min: dict[str, float] = {}
    has_gt = products is not None and products and "group_id" in products[0]
    if has_gt:
        for p in products:
            gid = p["group_id"]
            price = float(p["price"])
            id_to_group[p["id"]] = gid
            if gid not in group_to_min or price < group_to_min[gid]:
                group_to_min[gid] = price

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["canonical_name", "category", "min_price"]
        if has_gt:
            header.append("gt_min_price")
        header += ["best_deal_id", "best_deal_name", "duplicate_count", "all_variants"]
        writer.writerow(header)

        for rec in output:
            variants = " | ".join(
                f"{s['name']} (₪{s['price']:.0f})" for s in rec["sources"]
            )
            row = [
                rec["canonical_name"],
                rec["category"],
                f"{rec['min_price']:.2f}",
            ]
            if has_gt:
                # derive group from the first source id present in ground truth
                gt_price = ""
                for s in rec["sources"]:
                    gid = id_to_group.get(s["id"])
                    if gid and gid in group_to_min:
                        gt_price = f"{group_to_min[gid]:.2f}"
                        break
                row.append(gt_price)
            row += [
                rec["best_deal_id"],
                rec["best_deal_name"],
                rec["duplicate_count"],
                variants,
            ]
            writer.writerow(row)


# ── Evaluation ───────────────────────────────────────────────────────────────

def _to_pairs(clusters: list[set[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for cluster in clusters:
        ids = sorted(cluster)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    return pairs


def evaluate(output: list[dict], products: list[dict]) -> dict:
    """
    Comprehensive evaluation against ground-truth group_id labels.

    Metrics:
    1. Pair Precision / Recall / F1  — industry standard, unit = pair of products
    2. Cluster Purity  — penalises false merges
    3. Cluster Coverage  — penalises missed duplicates
    4. B-Cubed P/R/F1  — entity-level, handles singletons fairly
    5. Adjusted Rand Index  — global partition similarity, chance-corrected
    6. Price Accuracy  — did we find the true lowest price per GT group?
       price_accuracy:        % of multi-product GT groups where we found the true min-price vendor
       avg_price_overcharge:  average % above the true min price we would show (0% = perfect)
    """
    from sklearn.metrics import adjusted_rand_score

    id_to_product = {p["id"]: p for p in products}

    # ── GT structures ─────────────────────────────────────────────────────────
    gt_groups: dict[str, set[str]] = {}
    for p in products:
        gt_groups.setdefault(p["group_id"], set()).add(p["id"])

    id_to_gt_group: dict[str, set[str]] = {
        pid: group for group in gt_groups.values() for pid in group
    }

    # ── Predicted structures ──────────────────────────────────────────────────
    pred_clusters: list[set[str]] = [
        {s["id"] for s in record["sources"]} for record in output
    ]
    id_to_pred_cluster: dict[str, set[str]] = {
        pid: cluster for cluster in pred_clusters for pid in cluster
    }

    # ── 1. Pair-based P / R / F1 ─────────────────────────────────────────────
    gt_pairs   = _to_pairs(list(gt_groups.values()))
    pred_pairs = _to_pairs(pred_clusters)

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs  - pred_pairs)

    pair_p  = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    pair_r  = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    pair_f1 = 2 * pair_p * pair_r / (pair_p + pair_r) if (pair_p + pair_r) > 0 else 0.0

    # ── 2. Cluster Purity ─────────────────────────────────────────────────────
    purity_sum = 0.0
    for cluster in pred_clusters:
        gt_counts: dict[str, int] = {}
        for pid in cluster:
            g = id_to_product[pid]["group_id"]
            gt_counts[g] = gt_counts.get(g, 0) + 1
        purity_sum += max(gt_counts.values())
    purity = purity_sum / len(products)

    # ── 3. Cluster Coverage ───────────────────────────────────────────────────
    coverage_sum = 0.0
    for gt_group in gt_groups.values():
        pred_counts: dict[int, int] = {}
        for pid in gt_group:
            c_id = id(id_to_pred_cluster[pid])
            pred_counts[c_id] = pred_counts.get(c_id, 0) + 1
        coverage_sum += max(pred_counts.values())
    coverage = coverage_sum / len(products)

    # ── 4. B-Cubed ───────────────────────────────────────────────────────────
    b3_p_scores, b3_r_scores = [], []
    for pid in id_to_product:
        pred_c  = id_to_pred_cluster[pid]
        gt_g    = id_to_gt_group[pid]
        overlap = len(pred_c & gt_g)
        b3_p_scores.append(overlap / len(pred_c))
        b3_r_scores.append(overlap / len(gt_g))

    b3_p  = sum(b3_p_scores) / len(b3_p_scores)
    b3_r  = sum(b3_r_scores) / len(b3_r_scores)
    b3_f1 = 2 * b3_p * b3_r / (b3_p + b3_r) if (b3_p + b3_r) > 0 else 0.0

    # ── 5. Adjusted Rand Index ────────────────────────────────────────────────
    sorted_ids   = sorted(id_to_product.keys())
    labels_true  = [id_to_product[pid]["group_id"]   for pid in sorted_ids]
    labels_pred  = [str(id(id_to_pred_cluster[pid])) for pid in sorted_ids]
    ari = adjusted_rand_score(labels_true, labels_pred)

    # ── 6. Price accuracy ─────────────────────────────────────────────────────
    # For each GT group with >1 member: find the dominant predicted cluster
    # (the one containing the most members), then compare its min price to the
    # true minimum across the whole GT group.
    price_correct = 0
    price_errors  = []
    gt_groups_multi = {gid: g for gid, g in gt_groups.items() if len(g) > 1}

    for gt_group in gt_groups_multi.values():
        true_min = min(float(id_to_product[pid]["price"]) for pid in gt_group)

        # Find dominant predicted cluster for this GT group
        cluster_votes: dict[int, list[str]] = {}
        for pid in gt_group:
            ckey = id(id_to_pred_cluster[pid])
            cluster_votes.setdefault(ckey, []).append(pid)
        dominant_pids = max(cluster_votes.values(), key=len)
        pred_min = min(float(id_to_product[pid]["price"]) for pid in dominant_pids)

        error = (pred_min - true_min) / true_min if true_min > 0 else 0.0
        price_errors.append(error)
        if abs(error) < 1e-6:
            price_correct += 1

    price_accuracy       = price_correct / len(gt_groups_multi) if gt_groups_multi else 1.0
    avg_price_overcharge = sum(price_errors) / len(price_errors) if price_errors else 0.0

    # ── Error examples ────────────────────────────────────────────────────────
    def pair_examples(pairs, limit=3):
        out = []
        for a_id, b_id in list(pairs)[:limit]:
            a = id_to_product.get(a_id, {}).get("name", a_id)
            b = id_to_product.get(b_id, {}).get("name", b_id)
            out.append(f'"{a}"  vs  "{b}"')
        return out

    return {
        "pair_precision":         round(pair_p,  4),
        "pair_recall":            round(pair_r,  4),
        "pair_f1":                round(pair_f1, 4),
        "pair_tp": tp, "pair_fp": fp, "pair_fn": fn,
        "cluster_purity":         round(purity,   4),
        "cluster_coverage":       round(coverage, 4),
        "bcubed_precision":       round(b3_p,  4),
        "bcubed_recall":          round(b3_r,  4),
        "bcubed_f1":              round(b3_f1, 4),
        "adjusted_rand_index":    round(ari,   4),
        "price_accuracy":         round(price_accuracy,       4),
        "avg_price_overcharge":   round(avg_price_overcharge, 4),
        "false_positive_examples": pair_examples(pred_pairs - gt_pairs),
        "false_negative_examples": pair_examples(gt_pairs   - pred_pairs),
    }


def save_evaluation(m: dict, output_dir: str) -> str:
    """
    Save evaluation results:
      - evaluation.png  — metrics bar chart + TP/FP/FN + price accuracy
      - evaluation.json — all numeric metrics + FP/FN examples
    Returns the path to the PNG.
    """
    plt.rcParams.update({"font.family": "sans-serif"})
    fig = plt.figure(figsize=(16, 7), facecolor="#F8F9FA")
    fig.suptitle("Zap Deduplication — Evaluation Report", fontsize=15, fontweight="bold",
                 color="#212121", y=1.01)

    gs = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1], wspace=0.4)
    ax_metrics = fig.add_subplot(gs[0])
    ax_errors  = fig.add_subplot(gs[1])
    ax_price   = fig.add_subplot(gs[2])

    for ax in (ax_metrics, ax_errors, ax_price):
        ax.set_facecolor("#FFFFFF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#E0E0E0")

    # ── Left: horizontal bar chart ─────────────────────────────────────────
    # Groups: (label, value, color, group_name)
    rows = [
        ("Pair Precision",    m["pair_precision"],    "#42A5F5", "Pair-based"),
        ("Pair Recall",       m["pair_recall"],       "#42A5F5", None),
        ("Pair F1",           m["pair_f1"],           "#1565C0", None),
        (None, None, None, None),  # spacer
        ("Cluster Purity",   m["cluster_purity"],    "#66BB6A", "Cluster"),
        ("Cluster Coverage", m["cluster_coverage"],  "#66BB6A", None),
        (None, None, None, None),  # spacer
        ("B-Cubed Precision", m["bcubed_precision"], "#FFA726", "B-Cubed"),
        ("B-Cubed Recall",    m["bcubed_recall"],    "#FFA726", None),
        ("B-Cubed F1",        m["bcubed_f1"],        "#E65100", None),
        (None, None, None, None),  # spacer
        ("Adj. Rand Index",   m["adjusted_rand_index"], "#AB47BC", "Global"),
    ]

    y_pos    = 0
    yticks   = []
    ylabels  = []
    bar_ys   = []
    bar_vals = []
    bar_cols = []
    group_markers = []  # (y_mid, group_name) for group labels on the right

    current_group_ys = []
    current_group_name = None

    for label, val, color, group in rows:
        if label is None:
            # flush group
            if current_group_ys and current_group_name:
                group_markers.append((
                    (current_group_ys[0] + current_group_ys[-1]) / 2,
                    current_group_name
                ))
            current_group_ys = []
            current_group_name = None
            y_pos -= 0.4  # smaller spacer gap
            continue
        if group:
            current_group_name = group
        current_group_ys.append(y_pos)
        bar_ys.append(y_pos)
        bar_vals.append(min(val, 1.0))  # cap at 1 for bar width (purity can exceed 1)
        bar_cols.append(color)
        yticks.append(y_pos)
        ylabels.append(label)
        y_pos -= 1

    # flush last group
    if current_group_ys and current_group_name:
        group_markers.append((
            (current_group_ys[0] + current_group_ys[-1]) / 2,
            current_group_name
        ))

    bars = ax_metrics.barh(bar_ys, bar_vals, color=bar_cols,
                            edgecolor="none", height=0.62, zorder=3)

    # shaded group bands
    group_band_colors = {"Pair-based": "#E3F2FD", "Cluster": "#E8F5E9",
                         "B-Cubed": "#FFF3E0", "Global": "#F3E5F5"}
    group_band_ranges = {}
    idx = 0
    for label, val, color, group in rows:
        if label is None:
            continue
        gname = group if group else list(group_band_ranges)[-1] if group_band_ranges else None
        if gname:
            if gname not in group_band_ranges:
                group_band_ranges[gname] = [bar_ys[idx], bar_ys[idx]]
            else:
                group_band_ranges[gname][1] = bar_ys[idx]
        idx += 1

    for gname, (y_top, y_bot) in group_band_ranges.items():
        ax_metrics.axhspan(y_bot - 0.38, y_top + 0.38,
                           facecolor=group_band_colors.get(gname, "#FAFAFA"),
                           alpha=0.45, zorder=1)

    ax_metrics.axvline(1.0, color="#BDBDBD", linestyle="--", linewidth=0.9, zorder=2)
    ax_metrics.set_xlim(0, 1.16)
    ax_metrics.set_xlabel("Score", fontsize=10, color="#555555")
    ax_metrics.set_title("Metrics", fontsize=12, fontweight="bold", pad=10, color="#212121")
    ax_metrics.set_yticks(yticks)
    ax_metrics.set_yticklabels(ylabels, fontsize=10)
    ax_metrics.tick_params(axis="x", colors="#777777")
    ax_metrics.grid(axis="x", color="#F0F0F0", linewidth=0.8, zorder=0)
    ax_metrics.set_axisbelow(True)

    raw_vals = [
        m["pair_precision"], m["pair_recall"], m["pair_f1"],
        m["cluster_purity"], m["cluster_coverage"],
        m["bcubed_precision"], m["bcubed_recall"], m["bcubed_f1"],
        m["adjusted_rand_index"],
    ]
    for bar, val in zip(bars, raw_vals):
        ax_metrics.text(
            min(val, 1.0) + 0.013, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold", color="#212121",
        )

    # group labels on the right side
    gname_colors = {"Pair-based": "#1565C0", "Cluster": "#2E7D32",
                    "B-Cubed": "#E65100", "Global": "#6A1B9A"}
    for y_mid, gname in group_markers:
        ax_metrics.text(
            1.155, y_mid, gname,
            va="center", ha="right", fontsize=8, fontstyle="italic",
            color=gname_colors.get(gname, "#555555"),
        )

    # ── Middle: TP / FP / FN ──────────────────────────────────────────────
    tp, fp, fn = m["pair_tp"], m["pair_fp"], m["pair_fn"]
    bar_colors = ["#43A047", "#E53935", "#FB8C00"]
    rects = ax_errors.bar([0, 1, 2], [tp, fp, fn],
                           color=bar_colors, edgecolor="none", width=0.5, zorder=3)
    ax_errors.set_title("Pair Breakdown", fontsize=12, fontweight="bold", pad=10, color="#212121")
    ax_errors.set_ylabel("# Pairs", fontsize=10, color="#555555")
    ax_errors.set_xticks([0, 1, 2])
    ax_errors.set_xticklabels(["TP", "FP", "FN"], fontsize=11, fontweight="bold")
    ax_errors.tick_params(axis="x", colors="#555555")
    ax_errors.tick_params(axis="y", colors="#777777")
    ax_errors.grid(axis="y", color="#F0F0F0", linewidth=0.8, zorder=0)
    ax_errors.set_axisbelow(True)
    y_max = max(tp, fp, fn, 1)
    ax_errors.set_ylim(0, y_max * 1.22)
    for rect, val, c in zip(rects, [tp, fp, fn], bar_colors):
        ax_errors.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + y_max * 0.03,
            str(val), ha="center", va="bottom", fontweight="bold", fontsize=12, color=c,
        )

    # ── Right: Price accuracy ─────────────────────────────────────────────
    pa  = m["price_accuracy"]
    apo = m["avg_price_overcharge"]
    pr  = ax_price.bar([0, 1], [pa, apo],
                        color=["#00838F", "#EF6C00"], edgecolor="none", width=0.45, zorder=3)
    ax_price.axhline(1.0, color="#BDBDBD", linestyle="--", linewidth=0.9, zorder=2)
    ax_price.set_ylim(0, max(1.0, apo * 1.5) * 1.2)
    ax_price.set_title("Price Quality", fontsize=12, fontweight="bold", pad=10, color="#212121")
    ax_price.set_ylabel("Rate / Fraction", fontsize=10, color="#555555")
    ax_price.set_xticks([0, 1])
    ax_price.set_xticklabels(["Price\nAccuracy", "Avg\nOvercharge"], fontsize=10, fontweight="bold")
    ax_price.tick_params(axis="x", colors="#555555")
    ax_price.tick_params(axis="y", colors="#777777")
    ax_price.grid(axis="y", color="#F0F0F0", linewidth=0.8, zorder=0)
    ax_price.set_axisbelow(True)
    for rect, val, c in zip(pr, [pa, apo], ["#00838F", "#EF6C00"]):
        ax_price.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 0.015,
            f"{val:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=11, color=c,
        )

    plt.tight_layout()
    png_path = os.path.join(output_dir, "evaluation.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # ── JSON ──────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    return png_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_pipeline():
    print("=" * 60)
    print("  Zap Deduplication v3")
    print("=" * 60)

    products = load_csv(INPUT_CSV)
    categories = len(set(p["category"] for p in products))
    has_gt = "group_id" in products[0]
    print(f"\nLoaded {len(products)} products across {categories} categories")
    if has_gt:
        print("Ground-truth labels found (group_id) — evaluation will run.\n")

    # Load caches (skip API calls for already-seen names)
    print("── Caches ──")
    load_normalization_cache()
    load_embedding_cache()
    print()

    # Stage 0 + 1 + 2 — normalize Hebrew names, embed, cluster per category
    print("── Stage 0+1+2: Hebrew normalization → Embeddings → FAISS clustering ──")
    clusters = cluster_by_category(products)

    # Save both caches after all API calls are done
    save_normalization_cache()
    save_embedding_cache()
    print(f"\n   {len(clusters)} clusters | {sum(1 for c in clusters if len(c) > 1)} candidate groups")
    print(f"   {len(_normalization_cache)} normalizations, {len(_embedding_cache)} embeddings in cache\n")

    # Stage 3 — LLM refinement for non-singletons (receives original names)
    print("── Stage 3: LLM cluster refinement ──")
    output: list[dict] = []
    llm_calls = 0

    for cluster in clusters:
        if len(cluster) == 1:
            p = cluster[0]
            output.append(build_record(p["name"], cluster))
        else:
            refined = llm_refine(cluster)
            llm_calls += 1
            for group in refined:
                output.append(build_record(group["canonical_name"], group["products"]))
                if len(group["products"]) > 1:
                    print(f"   '{group['canonical_name']}' <- {len(group['products'])} listings")

    print(f"\n   {llm_calls} LLM calls\n")

    # Stage 4 — sort & save
    output.sort(key=lambda x: x["min_price"])
    os.makedirs("output", exist_ok=True)

    save_csv(output, OUTPUT_CSV, products if has_gt else None)

    dupes = sum(1 for x in output if x["duplicate_count"] > 1)
    saved = len(products) - len(output)
    print("=" * 60)
    print(f"  {len(output)} unique products (was {len(products)})")
    print(f"  {dupes} duplicate groups | {saved} redundant listings removed")
    print(f"  Output -> {OUTPUT_CSV}")

    if has_gt:
        metrics  = evaluate(output, products)
        png_path = save_evaluation(metrics, "output")
        print(f"  Evaluation -> {png_path}  |  evaluation.json")
        print(f"\n  Pair F1: {metrics['pair_f1']:.3f}  |  "
              f"Precision: {metrics['pair_precision']:.3f}  |  "
              f"Recall: {metrics['pair_recall']:.3f}")
        print(f"  Price accuracy: {metrics['price_accuracy']:.1%}  |  "
              f"Avg overcharge: {metrics['avg_price_overcharge']:.2%}")

    print("=" * 60 + "\n")
    return output


if __name__ == "__main__":
    run_pipeline()
