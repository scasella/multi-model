"""Reasoning-diversity analysis across variants on MATH500 rollouts.

Thesis being tested: "<multipersonaDebate> demonstrates better novelty,
discovery, and problem solving than pure <think>".

Operationalization — across the N samples drawn per problem, measure
how much the reasoning TRAJECTORIES vary. A scaffold that encourages
divergent exploration should produce higher-diversity samples than one
that converges on a single strategy, even when both reach the same
answer.

Pipeline
--------
1. Load rollouts.jsonl for each variant; group by unique_id.
2. For each sample, extract the reasoning trace (inside <think>,
   <mutipersonaDebate>, or <answer> — everything that's NOT the final
   boxed answer).
3. Embed each trace with sentence-transformers (`all-mpnet-base-v2`);
   truncate to 512 tokens of the trace head — the OPENING determines
   which strategy was chosen, which is what we actually want.
4. Per problem, across the N samples, compute:
     - mean pairwise cosine DISTANCE (1 - similarity) → diversity score
     - trace of covariance on embeddings → spread
     - DBSCAN cluster count at eps chosen from global distance distn
       → "distinct strategies"
     - completion-length spread (sanity check: is diversity just length?)
5. Aggregate by level, subject, variant.
6. Paired comparison (same problems, panel vs thinking).

Usage
-----
    python scripts/analyze_diversity.py \\
        inputs='{"panel":"reports/eval_math500_vibecheck/math_final_panel/rollouts.jsonl",\\
                 "thinking":"reports/eval_math500_vibecheck/thinking_native/rollouts.jsonl"}' \\
        out_dir=reports/diversity_analysis
"""
from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import chz
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("analyze_diversity")


_PANEL_RE = re.compile(r"<mutipersonaDebate>(.*?)(?:</mutipersonaDebate>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_BOXED_TRAIL_RE = re.compile(r"\\boxed\{.*$", re.DOTALL)


def extract_reasoning(completion: str) -> str:
    """Extract the reasoning trace from a completion.

    Priority order:
    1. <mutipersonaDebate>...</mutipersonaDebate> body (panel variant)
    2. <think>...</think> body (thinking variant)
    3. Full completion with any trailing \\boxed{…} stripped
       (fallback — covers no_panel, instruct, and any bad-formatted traces)
    """
    m = _PANEL_RE.search(completion)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = _THINK_RE.search(completion)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Fallback — strip trailing \boxed{…} so we're left with the reasoning
    stripped = _BOXED_TRAIL_RE.sub("", completion).strip()
    return stripped if stripped else completion


def load_variant(path: Path) -> dict[str, list[dict]]:
    """Load a rollouts.jsonl → {unique_id: [sample_records, ...]}.

    Falls back to synthesized id `{year}_{problem_idx}` for AIME rollouts
    (which don't carry a `unique_id` field).
    """
    by_problem: dict[str, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["reasoning"] = extract_reasoning(r.get("completion", ""))
            uid = r.get("unique_id")
            if uid is None and "year" in r and "problem_idx" in r:
                uid = f"aime{r['year']}_{r['problem_idx']}"
                r["unique_id"] = uid
            by_problem[uid].append(r)
    # Sort each group by sample_idx for determinism.
    for uid in by_problem:
        by_problem[uid].sort(key=lambda r: r.get("sample_idx", 0))
    return dict(by_problem)


def pairwise_cos_dist(embs: np.ndarray) -> np.ndarray:
    """N×N cosine-distance matrix (1 - cos_sim). Zero diagonal."""
    # Rows assumed normalized by sentence-transformers already.
    sim = embs @ embs.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def mean_offdiag(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    return float(mat[iu].mean())


@chz.chz
class AnalyzeConfig:
    # JSON dict: {label: path-to-rollouts.jsonl}
    inputs: str = json.dumps({
        "panel_postmath":   "reports/eval_math500_vibecheck/math_final_panel/rollouts.jsonl",
        "thinking":         "reports/eval_math500_vibecheck/thinking_native/rollouts.jsonl",
    })
    out_dir: str = str(ROOT / "reports" / "diversity_analysis")
    # Embedding model — all-mpnet-base-v2 is a good small default (384-d after
    # normalization, 512-token limit). MiniLM is faster if you want speed.
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    # Trace length: empirically the OPENING of a reasoning trace determines
    # which strategy is taken. Truncating to ~first N chars keeps the signal
    # focused on strategy choice rather than late-trace verbosity.
    trace_head_chars: int = 2000
    # DBSCAN eps. Smaller → more, tighter clusters.
    dbscan_eps: float = 0.25
    dbscan_min_samples: int = 2


def _embed_variant(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    embs = m.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs


def analyze(cfg: AnalyzeConfig) -> None:
    inputs = json.loads(cfg.inputs)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("variants: %s", list(inputs.keys()))
    logger.info("out_dir: %s", out_dir)

    # --- load all variants ---
    variants: dict[str, dict[str, list[dict]]] = {}
    for label, path in inputs.items():
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            logger.warning("%s: %s does not exist — skipping", label, p)
            continue
        data = load_variant(p)
        logger.info("%s: %d problems, %d total samples from %s",
                    label, len(data), sum(len(v) for v in data.values()), p)
        variants[label] = data

    if not variants:
        logger.error("No variants loaded.")
        return

    # --- intersect problem set across variants (for paired comparison) ---
    shared_uids = set.intersection(*(set(v.keys()) for v in variants.values()))
    logger.info("shared problems across all variants: %d", len(shared_uids))

    # --- flatten for a single embedding pass ---
    # records: (label, uid, sample_idx, text_for_embed, correct, level, subject, completion_tokens)
    records: list[dict] = []
    for label, data in variants.items():
        for uid, samples in data.items():
            for s in samples:
                head = s["reasoning"][: cfg.trace_head_chars]
                records.append({
                    "label": label,
                    "uid": uid,
                    "sample_idx": s.get("sample_idx", 0),
                    "text": head,
                    "correct": bool(s.get("correct", False)),
                    "level": str(s.get("level", "?")).strip(),
                    "subject": s.get("subject", "?"),
                    "completion_tokens": int(s.get("completion_tokens", 0)),
                })
    logger.info("embedding %d traces with %s", len(records), cfg.embed_model)
    texts = [r["text"] for r in records]
    embs = _embed_variant(texts, cfg.embed_model)
    logger.info("embedding shape: %s", embs.shape)

    # Attach embedding indices back to records.
    for i, r in enumerate(records):
        r["_idx"] = i

    # --- per-problem metrics per variant ---
    from sklearn.cluster import DBSCAN

    per_problem: list[dict] = []
    for label in variants:
        for uid in sorted(variants[label].keys()):
            samples = [r for r in records if r["label"] == label and r["uid"] == uid]
            if not samples:
                continue
            idx = np.array([r["_idx"] for r in samples])
            E = embs[idx]
            D = pairwise_cos_dist(E)
            mean_pw = mean_offdiag(D)

            # Variance across samples = trace of covariance of (normalized) embeddings
            centered = E - E.mean(axis=0, keepdims=True)
            var_trace = float(np.mean(np.sum(centered ** 2, axis=1)))  # mean squared deviation

            # Strategy clusters via DBSCAN on the embeddings
            if E.shape[0] >= cfg.dbscan_min_samples:
                db = DBSCAN(
                    eps=cfg.dbscan_eps,
                    min_samples=cfg.dbscan_min_samples,
                    metric="cosine",
                ).fit(E)
                labels = db.labels_
                n_clust = len({c for c in labels if c != -1})
                n_noise = int((labels == -1).sum())
                # Count "distinct strategies" = clusters + noise points (each a singleton strategy)
                distinct = n_clust + n_noise
            else:
                distinct = E.shape[0]

            lens = np.array([r["completion_tokens"] for r in samples])
            corr = np.array([r["correct"] for r in samples])
            per_problem.append({
                "label": label,
                "uid": uid,
                "level": samples[0]["level"],
                "subject": samples[0]["subject"],
                "n_samples": int(E.shape[0]),
                "n_correct": int(corr.sum()),
                "pass_at_n": bool(corr.any()),
                "mean_pairwise_dist": mean_pw,
                "emb_variance_trace": var_trace,
                "distinct_strategies": int(distinct),
                "len_mean": float(lens.mean()),
                "len_std": float(lens.std()),
                "len_cv": float(lens.std() / max(lens.mean(), 1)),
            })

    # --- per-variant aggregates ---
    by_variant: dict[str, dict] = {}
    for label in variants:
        recs = [p for p in per_problem if p["label"] == label]
        if not recs:
            continue
        arr = lambda k: np.array([r[k] for r in recs])
        by_variant[label] = {
            "n_problems": len(recs),
            "total_samples": int(sum(r["n_samples"] for r in recs)),
            "mean_pairwise_dist_mean": float(arr("mean_pairwise_dist").mean()),
            "mean_pairwise_dist_std": float(arr("mean_pairwise_dist").std()),
            "distinct_strategies_mean": float(arr("distinct_strategies").mean()),
            "distinct_strategies_std": float(arr("distinct_strategies").std()),
            "len_mean_mean": float(arr("len_mean").mean()),
            "len_cv_mean": float(arr("len_cv").mean()),
            "emb_variance_trace_mean": float(arr("emb_variance_trace").mean()),
            "pass_at_n_rate": float(np.mean([r["pass_at_n"] for r in recs])),
        }

    # --- paired comparison on shared problems ---
    paired: dict[str, dict] = {}
    labels = list(variants.keys())
    if len(labels) >= 2 and shared_uids:
        idx_map = {(p["label"], p["uid"]): p for p in per_problem}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                diffs_mpd, diffs_ds = [], []
                for uid in sorted(shared_uids):
                    pa = idx_map.get((a, uid))
                    pb = idx_map.get((b, uid))
                    if pa and pb:
                        diffs_mpd.append(pa["mean_pairwise_dist"] - pb["mean_pairwise_dist"])
                        diffs_ds.append(pa["distinct_strategies"] - pb["distinct_strategies"])
                if diffs_mpd:
                    mpd = np.array(diffs_mpd)
                    ds = np.array(diffs_ds)
                    # paired t-test (lightweight — no scipy dep)
                    def _t_p(x):
                        n = len(x)
                        if n < 2 or x.std() == 0:
                            return None, None
                        t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
                        # rough 2-sided p approximation via normal
                        from math import erf, sqrt
                        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
                        return float(t), float(p)
                    t_mpd, p_mpd = _t_p(mpd)
                    t_ds, p_ds = _t_p(ds)
                    paired[f"{a}_minus_{b}"] = {
                        "n_paired": int(len(diffs_mpd)),
                        "mean_pairwise_dist_delta_mean": float(mpd.mean()),
                        "mean_pairwise_dist_delta_std": float(mpd.std()),
                        "mean_pairwise_dist_t": t_mpd,
                        "mean_pairwise_dist_p_approx": p_mpd,
                        "distinct_strategies_delta_mean": float(ds.mean()),
                        "distinct_strategies_t": t_ds,
                        "distinct_strategies_p_approx": p_ds,
                    }

    # --- by level per variant ---
    by_level: dict[str, dict[str, dict]] = defaultdict(dict)
    for label in variants:
        recs = [p for p in per_problem if p["label"] == label]
        for lv in sorted({r["level"] for r in recs}):
            lv_recs = [r for r in recs if r["level"] == lv]
            if not lv_recs:
                continue
            by_level[label][lv] = {
                "n_problems": len(lv_recs),
                "mean_pairwise_dist": float(np.mean([r["mean_pairwise_dist"] for r in lv_recs])),
                "distinct_strategies": float(np.mean([r["distinct_strategies"] for r in lv_recs])),
                "pass_at_n": float(np.mean([r["pass_at_n"] for r in lv_recs])),
            }

    # --- write outputs ---
    out = {
        "config": {
            "inputs": inputs,
            "embed_model": cfg.embed_model,
            "trace_head_chars": cfg.trace_head_chars,
            "dbscan_eps": cfg.dbscan_eps,
        },
        "by_variant": by_variant,
        "by_level": dict(by_level),
        "paired": paired,
        "n_shared_problems": len(shared_uids),
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / "per_problem.jsonl", "w") as fh:
        for p in per_problem:
            fh.write(json.dumps(p) + "\n")
    logger.info("wrote %s", out_dir / "summary.json")

    # --- pretty print ---
    print("\n========== DIVERSITY SUMMARY ==========")
    print("variants:", list(variants.keys()))
    print(f"shared problems: {len(shared_uids)}")
    print("\nPer-variant (mean across problems):")
    hdr = ("variant", "n_probs", "mean_pw_dist", "dist_strat", "pass@n", "len_mean")
    print("  " + "  ".join(f"{h:>18s}" for h in hdr))
    for label, v in by_variant.items():
        vals = (
            label, v["n_problems"], v["mean_pairwise_dist_mean"],
            v["distinct_strategies_mean"], v["pass_at_n_rate"], v["len_mean_mean"],
        )
        print("  " + "  ".join(f"{x:>18}" if isinstance(x, (str,int)) else f"{x:>18.4f}" for x in vals))
    if paired:
        print("\nPaired comparisons (A - B, on shared problems):")
        for key, v in paired.items():
            print(f"  {key:30s}  n={v['n_paired']:3d}  "
                  f"ΔmeanPWdist={v['mean_pairwise_dist_delta_mean']:+.4f}  "
                  f"t={v['mean_pairwise_dist_t']:+.2f}  p≈{v['mean_pairwise_dist_p_approx']:.4f}  "
                  f"Δdistinct={v['distinct_strategies_delta_mean']:+.2f}")


if __name__ == "__main__":
    cfg = chz.entrypoint(AnalyzeConfig)
    analyze(cfg)
