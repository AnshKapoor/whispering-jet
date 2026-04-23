"""Generate HDBSCAN diagnostics for an experiment config or resolved experiment.

This script rebuilds the clustering input per flow, fits HDBSCAN with the
configured parameters, and writes diagnostic artefacts that help judge whether
the chosen `min_cluster_size`, `min_samples`, and cluster-selection settings are
reasonable.

Outputs per flow:
- condensed tree plot
- single-linkage tree plot
- minimum-spanning-tree plot (optional for smaller flows)
- cluster-persistence bar chart
- membership-probability histogram
- outlier-score histogram
- per-flow summary CSV and JSON metadata

Typical usage:
  python scripts/plot_hdbscan_diagnostics.py --experiment EXP016
  python scripts/plot_hdbscan_diagnostics.py --experiment EXP041 --max-flights-per-flow 1200
  python scripts/plot_hdbscan_diagnostics.py --config output/experiments/EXP016/config_resolved.yaml --flows Landung_27R,Start_09L
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import issparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering.distances import build_feature_matrix, pairwise_distance_matrix


PRECOMPUTED_METRICS = {"dtw", "frechet", "lcss", "euclidean_weighted"}
MAX_DIRECT_SINGLE_LINKAGE_MERGES = 1500


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_config(args: argparse.Namespace) -> tuple[str, Path, dict]:
    if args.config:
        cfg_path = Path(args.config)
        cfg = _load_yaml(cfg_path)
        exp_name = str((cfg.get("output", {}) or {}).get("experiment_name") or cfg_path.stem)
        return exp_name, cfg_path, cfg
    if not args.experiment:
        raise ValueError("Provide either --experiment or --config.")
    exp = str(args.experiment).strip()
    cfg_path = ROOT / "output" / "experiments" / exp / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Resolved config not found for {exp}: {cfg_path}")
    return exp, cfg_path, _load_yaml(cfg_path)


def _filter_flows(df: pd.DataFrame, flow_keys: Sequence[str], include: Sequence[Sequence[str]] | None) -> pd.DataFrame:
    if not include or not flow_keys:
        return df
    include_set = {tuple(item) for item in include}
    mask = df[list(flow_keys)].apply(tuple, axis=1).isin(include_set)
    return df.loc[mask].reset_index(drop=True)


def _parse_flows_arg(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    vals = {item.strip() for item in str(raw).split(",") if item.strip()}
    return vals or None


def _sample_flow_ids(flow_df: pd.DataFrame, sample_n: int, rng: np.random.Generator) -> list[int]:
    ids = np.sort(flow_df["flight_id"].drop_duplicates().to_numpy(dtype=int))
    if ids.size <= sample_n:
        return ids.tolist()
    sampled = rng.choice(ids, size=sample_n, replace=False)
    return np.sort(sampled).tolist()


def _sanitize_dense_precomputed(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.isfinite(D).all():
        n_bad = int((~np.isfinite(D)).sum())
        raise ValueError(f"Distance matrix contains non-finite values (count={n_bad}).")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _fit_hdbscan(X_fit, *, precomputed: bool, cluster_params: dict):
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("hdbscan is required. Install via `pip install hdbscan`.") from exc

    params = dict(cluster_params or {})
    params["gen_min_span_tree"] = True
    params["prediction_data"] = True
    if precomputed:
        params["metric"] = "precomputed"
    model = hdbscan.HDBSCAN(**params)
    labels = model.fit_predict(X_fit)
    return model, np.asarray(labels, dtype=int)


def _save_condensed_tree_plot(model, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    model.condensed_tree_.plot(select_clusters=True, axis=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_single_linkage_plot(model, title: str, path: Path) -> None:
    tree_df = model.single_linkage_tree_.to_pandas()
    n_merges = int(len(tree_df))

    if n_merges <= MAX_DIRECT_SINGLE_LINKAGE_MERGES:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            model.single_linkage_tree_.plot(axis=ax)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            return
        except RecursionError:
            pass

    fig, ax = plt.subplots(figsize=(10, 6))
    merge_idx = np.arange(n_merges, dtype=int)
    distances = tree_df["distance"].to_numpy(dtype=float)
    sizes = tree_df["size"].to_numpy(dtype=float)

    # Large trees can overflow SciPy's recursive dendrogram code; summarize the
    # merge structure instead by plotting merge distance against merge order.
    sc = ax.scatter(
        merge_idx,
        distances,
        c=np.log10(np.maximum(sizes, 1.0)),
        s=12,
        cmap="viridis",
        alpha=0.85,
        edgecolors="none",
    )
    ax.plot(merge_idx, distances, color="#4C78A8", alpha=0.35, linewidth=1.0)
    ax.set_title(f"{title} (merge-distance summary)")
    ax.set_xlabel("merge index")
    ax.set_ylabel("mutual reachability distance")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(merge size)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_mst_plot(model, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    model.minimum_spanning_tree_.plot(axis=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_hist(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.asarray(values, dtype=float), bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_persistence_bar(values: np.ndarray, title: str, path: Path) -> None:
    vals = np.asarray(values, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(8, 5))
    if vals.size:
        ax.bar(np.arange(vals.size), vals, color="#F58518")
    ax.set_title(title)
    ax.set_xlabel("selected cluster index")
    ax.set_ylabel("cluster persistence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _labels_summary(labels: np.ndarray) -> tuple[int, float]:
    unique = [int(c) for c in np.unique(labels) if c != -1]
    noise_frac = float(np.mean(labels == -1)) if labels.size else 0.0
    return len(unique), noise_frac


def _flow_name(flow_keys: Sequence[str], flow_vals: tuple) -> str:
    return "_".join(str(v) for v in flow_vals) if flow_keys else "GLOBAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HDBSCAN diagnostics for an experiment.")
    parser.add_argument("--experiment", help="Experiment name, e.g. EXP016.")
    parser.add_argument("--config", help="Path to a config_resolved.yaml or experiment YAML.")
    parser.add_argument(
        "--flows",
        help="Optional comma-separated flow labels to keep, e.g. Landung_27R,Start_09L",
    )
    parser.add_argument(
        "--max-flights-per-flow",
        type=int,
        default=None,
        help="Optional cap to sample flights per flow for lighter diagnostics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=11,
        help="Random seed for optional flight sampling.",
    )
    parser.add_argument(
        "--densify-sparse-max-n",
        type=int,
        default=2000,
        help="Densify sparse precomputed matrices only when n is at most this threshold.",
    )
    parser.add_argument(
        "--skip-mst-above",
        type=int,
        default=500,
        help="Skip minimum-spanning-tree plot when n exceeds this threshold.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory (default: output/eda/hdbscan_diagnostics/<EXP>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_name, cfg_path, cfg = _resolve_config(args)

    clustering_cfg = cfg.get("clustering", {}) or {}
    method = str(clustering_cfg.get("method", "")).strip().lower()
    if method != "hdbscan":
        raise ValueError(f"Config method is {method!r}; this script only supports HDBSCAN experiments.")

    cluster_params = dict(clustering_cfg.get("hdbscan", {}) or {})
    distance_metric = str(clustering_cfg.get("distance_metric", "euclidean")).strip().lower()
    distance_params = dict(clustering_cfg.get("distance_params", {}) or {})
    preprocessed_csv = (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if not preprocessed_csv:
        raise ValueError("Config is missing input.preprocessed_csv.")

    flow_keys = list((cfg.get("flows", {}) or {}).get("flow_keys") or [])
    include_flows = (cfg.get("flows", {}) or {}).get("include", []) or []
    vector_cols = list((cfg.get("features", {}) or {}).get("vector_cols") or ["x_utm", "y_utm"])
    selected_flows = _parse_flows_arg(args.flows)

    preprocessed_path = Path(preprocessed_csv)
    if not preprocessed_path.is_absolute():
        preprocessed_path = (ROOT / preprocessed_path).resolve()
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    outdir = Path(args.outdir) if args.outdir else (ROOT / "output" / "eda" / "hdbscan_diagnostics" / exp_name)
    outdir.mkdir(parents=True, exist_ok=True)

    usecols = ["flight_id", "step", *vector_cols, *flow_keys]
    df = pd.read_csv(preprocessed_path, usecols=usecols)
    df = _filter_flows(df, flow_keys, include_flows)

    rng = np.random.default_rng(int(args.random_state))
    precomputed = distance_metric in PRECOMPUTED_METRICS
    rows: list[dict] = []

    for flow_vals, flow_df in df.groupby(flow_keys, sort=True):
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_label = _flow_name(flow_keys, flow_vals)
        if selected_flows and flow_label not in selected_flows:
            continue

        flow_out = outdir / flow_label
        flow_out.mkdir(parents=True, exist_ok=True)

        total_flights = int(flow_df["flight_id"].nunique())
        fit_df = flow_df
        fit_mode = "full"
        if args.max_flights_per_flow and total_flights > int(args.max_flights_per_flow):
            keep_ids = _sample_flow_ids(flow_df, int(args.max_flights_per_flow), rng)
            fit_df = flow_df[flow_df["flight_id"].isin(keep_ids)].copy()
            fit_mode = f"sample_{len(keep_ids)}"

        X, trajs = build_feature_matrix(fit_df, vector_cols=vector_cols, allow_ragged=True)
        n_fit = int(X.shape[0])
        if n_fit < 3:
            rows.append(
                {
                    "flow": flow_label,
                    "n_flights_total": total_flights,
                    "n_flights_fit": n_fit,
                    "fit_mode": fit_mode,
                    "status": "skipped_too_small",
                }
            )
            continue

        note = ""
        if precomputed:
            D = pairwise_distance_matrix(
                trajs if distance_metric in {"dtw", "frechet", "lcss"} else X,
                metric=distance_metric,
                cache_dir=flow_out / "cache",
                flow_name=flow_label,
                params=distance_params,
                cache_ids=sorted(fit_df["flight_id"].drop_duplicates().tolist()),
            )
            if issparse(D):
                if n_fit <= int(args.densify_sparse_max_n):
                    D = _sanitize_dense_precomputed(D.toarray())
                    note = "densified_from_sparse"
                else:
                    rows.append(
                        {
                            "flow": flow_label,
                            "n_flights_total": total_flights,
                            "n_flights_fit": n_fit,
                            "fit_mode": fit_mode,
                            "status": "skipped_sparse_too_large",
                        }
                    )
                    continue
            else:
                D = _sanitize_dense_precomputed(D)
            fit_obj = D
        else:
            fit_obj = np.asarray(X, dtype=float)

        model, labels = _fit_hdbscan(fit_obj, precomputed=precomputed, cluster_params=cluster_params)
        n_clusters, noise_frac = _labels_summary(labels)
        persistence = np.asarray(getattr(model, "cluster_persistence_", np.array([])), dtype=float)
        probabilities = np.asarray(getattr(model, "probabilities_", np.array([])), dtype=float)
        outliers = np.asarray(getattr(model, "outlier_scores_", np.array([])), dtype=float)

        # Save flow-level plots.
        plot_notes: list[str] = []
        try:
            _save_condensed_tree_plot(
                model,
                title=f"{exp_name} {flow_label} condensed tree",
                path=flow_out / "condensed_tree.png",
            )
        except Exception as exc:
            plot_notes.append(f"condensed_tree_failed:{type(exc).__name__}")
        try:
            _save_single_linkage_plot(
                model,
                title=f"{exp_name} {flow_label} single linkage tree",
                path=flow_out / "single_linkage_tree.png",
            )
        except Exception as exc:
            plot_notes.append(f"single_linkage_failed:{type(exc).__name__}")
        if n_fit <= int(args.skip_mst_above):
            try:
                _save_mst_plot(
                    model,
                    title=f"{exp_name} {flow_label} mutual-reachability MST",
                    path=flow_out / "minimum_spanning_tree.png",
                )
            except Exception as exc:
                plot_notes.append(f"mst_failed:{type(exc).__name__}")
        _save_persistence_bar(
            persistence,
            title=f"{exp_name} {flow_label} cluster persistence",
            path=flow_out / "cluster_persistence.png",
        )
        _save_hist(
            probabilities,
            title=f"{exp_name} {flow_label} membership probabilities",
            xlabel="probability",
            path=flow_out / "membership_probabilities.png",
        )
        _save_hist(
            outliers,
            title=f"{exp_name} {flow_label} outlier scores",
            xlabel="outlier score",
            path=flow_out / "outlier_scores.png",
        )

        pd.DataFrame(
            {
                "label": labels,
                "probability": probabilities,
                "outlier_score": outliers,
            }
        ).to_csv(flow_out / "labels_probabilities.csv", index=False)
        pd.DataFrame({"cluster_persistence": persistence}).to_csv(flow_out / "cluster_persistence.csv", index=False)

        combined_note = note
        if plot_notes:
            combined_note = ";".join(filter(None, [combined_note, *plot_notes]))

        flow_meta = {
            "experiment": exp_name,
            "config": str(cfg_path),
            "flow": flow_label,
            "distance_metric": distance_metric,
            "cluster_params": cluster_params,
            "distance_params": distance_params,
            "precomputed": precomputed,
            "n_flights_total": total_flights,
            "n_flights_fit": n_fit,
            "fit_mode": fit_mode,
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "mean_cluster_persistence": float(np.mean(persistence)) if persistence.size else float("nan"),
            "max_cluster_persistence": float(np.max(persistence)) if persistence.size else float("nan"),
            "mean_membership_probability": float(np.mean(probabilities)) if probabilities.size else float("nan"),
            "median_membership_probability": float(np.median(probabilities)) if probabilities.size else float("nan"),
            "mean_outlier_score": float(np.mean(outliers)) if outliers.size else float("nan"),
            "max_outlier_score": float(np.max(outliers)) if outliers.size else float("nan"),
            "note": combined_note,
        }
        (flow_out / "metadata.json").write_text(json.dumps(flow_meta, indent=2), encoding="utf-8")
        rows.append({"status": "ok", **flow_meta})

    if not rows:
        raise RuntimeError("No flows processed. Check --flows, config include filters, or input file.")

    summary = pd.DataFrame(rows).sort_values(["status", "flow"]).reset_index(drop=True)
    summary.to_csv(outdir / "flow_diagnostics_summary.csv", index=False)
    metadata = {
        "experiment": exp_name,
        "config": str(cfg_path),
        "preprocessed_csv": str(preprocessed_path),
        "distance_metric": distance_metric,
        "cluster_params": cluster_params,
        "distance_params": distance_params,
        "selected_flows": sorted(selected_flows) if selected_flows else None,
        "max_flights_per_flow": args.max_flights_per_flow,
    }
    (outdir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {outdir / 'flow_diagnostics_summary.csv'}", flush=True)
    print(f"Wrote {outdir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
