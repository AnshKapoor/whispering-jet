"""Summarize noise-simulation vs clustering performance for EXP001-EXP062.

Inputs:
  - output/experiments/EXP###/{config_resolved.yaml,metrics_global.csv,labels_ALL.csv}
  - noise_simulation/results/EXP###/{summary_mse.csv,aggregate_totals/overall_summary.json}
  - noise_simulation/results/EXP###/aggregate_totals/category_summary.csv

Outputs:
  - output/eda/exp001_062_noise_clustering_experiment_summary.csv
  - output/eda/exp001_062_noise_clustering_flow_summary.csv
  - output/eda/exp001_062_noise_clustering_block_summary.csv
  - output/eda/exp001_062_noise_clustering_method_summary.csv
  - output/eda/exp001_062_noise_clustering_correlations.csv
  - thesis/docs/noise_clustering_results_exp001_062.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]


def _exp_name(exp_num: int) -> str:
    return f"EXP{exp_num:03d}"


def _exp_block(exp_num: int) -> tuple[str, str]:
    """Return thesis-friendly experiment block labels for EXP001-EXP062."""

    if 1 <= exp_num <= 10:
        return "EXP001-010", "Preprocessing variants"
    if 11 <= exp_num <= 15:
        return "EXP011-015", "KMeans baselines"
    if 16 <= exp_num <= 20:
        return "EXP016-020", "HDBSCAN baselines"
    if 21 <= exp_num <= 24:
        return "EXP021-024", "Non-Euclidean pilots"
    if 25 <= exp_num <= 31:
        return "EXP025-031", "OPTICS sensitivity"
    if 32 <= exp_num <= 36:
        return "EXP032-036", "DBSCAN sensitivity"
    if 37 <= exp_num <= 41:
        return "EXP037-041", "GMM selection"
    if 42 <= exp_num <= 46:
        return "EXP042-046", "Agglomerative and Birch"
    if 47 <= exp_num <= 50:
        return "EXP047-050", "3D altitude features"
    if 51 <= exp_num <= 57:
        return "EXP051-057", "DTW extension"
    if 58 <= exp_num <= 62:
        return "EXP058-062", "LCSS extension"
    return "OTHER", "Other"


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_preprocessed_id(raw: str | None) -> str | None:
    if not raw:
        return None
    match = re.search(r"preprocessed_(\d+)", str(raw))
    return match.group(1) if match else None


def _float_or_nan(value: object) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _scalar_from_csv(path: Path) -> dict[str, float]:
    df = _read_csv(path)
    if df.empty:
        return {}
    return {str(col): _float_or_nan(df.iloc[0][col]) for col in df.columns}


def _labels_summary(labels_path: Path) -> dict[str, float]:
    """Compute label-level counts from labels_ALL.csv.

    Output keys:
      total_flights, total_noise_flights, total_clustered_flights, noise_frac,
      clustered_frac, n_flows_total_labels, flows_n_clusters_gt1, n_clusters_total.
    """

    df = _read_csv(labels_path)
    if df.empty:
        return {}

    total_flights = int(len(df))
    cluster_ids = pd.to_numeric(df.get("cluster_id"), errors="coerce")
    noise_mask = cluster_ids.eq(-1)
    total_noise = int(noise_mask.sum())
    total_clustered = int((~noise_mask).sum())

    flow_col = "flow_label" if "flow_label" in df.columns else None
    n_flows_total = 0
    flows_n_clusters_gt1 = 0
    n_clusters_total = 0
    if flow_col:
        per_flow_clusters = (
            df.loc[~noise_mask, [flow_col, "cluster_id"]]
            .drop_duplicates()
            .groupby(flow_col)["cluster_id"]
            .nunique()
        )
        n_flows_total = int(df[flow_col].nunique())
        flows_n_clusters_gt1 = int((per_flow_clusters > 1).sum())
        n_clusters_total = int(per_flow_clusters.sum())

    return {
        "total_flights": total_flights,
        "total_noise_flights": total_noise,
        "total_clustered_flights": total_clustered,
        "noise_frac": total_noise / total_flights if total_flights else float("nan"),
        "clustered_frac": total_clustered / total_flights if total_flights else float("nan"),
        "n_flows_total_labels": n_flows_total,
        "flows_n_clusters_gt1": flows_n_clusters_gt1,
        "n_clusters_total": n_clusters_total,
    }


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values_num = pd.to_numeric(values, errors="coerce")
    weights_num = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid_mask = values_num.notna() & weights_num.gt(0)
    if not valid_mask.any():
        return float("nan")
    return float(np.average(values_num[valid_mask], weights=weights_num[valid_mask]))


def _noise_flow_summary(summary_mse_path: Path, category_summary_path: Path) -> pd.DataFrame:
    """Aggregate noise error per flow using aircraft-type flight counts as weights.

    Output columns:
      flow_label, A/D, Runway, total_flights, weighted_mae, weighted_mse, weighted_rmse
    """

    summary_df = _read_csv(summary_mse_path)
    category_df = _read_csv(category_summary_path)
    if summary_df.empty or category_df.empty:
        return pd.DataFrame()

    flight_counts = (
        summary_df.groupby(["A/D", "Runway", "aircraft_type"], as_index=False)["n_flights"].sum()
    )
    merged = category_df.merge(
        flight_counts,
        on=["A/D", "Runway", "aircraft_type"],
        how="left",
    )
    merged["n_flights"] = pd.to_numeric(merged["n_flights"], errors="coerce").fillna(0.0)
    if merged["n_flights"].sum() <= 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (ad, runway), group in merged.groupby(["A/D", "Runway"], dropna=False):
        rows.append(
            {
                "flow_label": f"{ad}_{runway}",
                "A/D": ad,
                "Runway": runway,
                "total_flights": float(group["n_flights"].sum()),
                "weighted_mae": _safe_weighted_mean(group["mae_cumulative_res"], group["n_flights"]),
                "weighted_mse": _safe_weighted_mean(group["mse_cumulative_res"], group["n_flights"]),
                "weighted_rmse": _safe_weighted_mean(group["rmse_cumulative_res"], group["n_flights"]),
                "n_aircraft_categories": int(group["aircraft_type"].nunique()),
            }
        )

    return pd.DataFrame(rows).sort_values(["A/D", "Runway"]).reset_index(drop=True)


def _status_for_noise_results(results_dir: Path) -> str:
    overall_path = results_dir / "aggregate_totals" / "overall_summary.json"
    category_path = results_dir / "aggregate_totals" / "category_summary.csv"
    summary_path = results_dir / "summary_mse.csv"
    if not results_dir.exists():
        return "missing_results_dir"
    if not summary_path.exists():
        return "missing_summary_mse"
    if not overall_path.exists():
        return "missing_overall_summary"
    if not category_path.exists():
        return "missing_category_summary"
    return "ok"


def _best_and_worst(
    df: pd.DataFrame,
    value_col: str,
    lower_is_better: bool = True,
) -> tuple[str | None, str | None]:
    valid = df[["exp", value_col]].dropna()
    if valid.empty:
        return None, None
    valid = valid.sort_values(value_col, ascending=lower_is_better)
    return str(valid.iloc[0]["exp"]), str(valid.iloc[-1]["exp"])


def _group_summary(experiments_df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in experiments_df.groupby(list(group_cols), dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = {col: key_tuple[idx] for idx, col in enumerate(group_cols)}
        valid_noise = group[group["noise_results_status"].eq("ok")].copy()

        best_all, worst_all = _best_and_worst(valid_noise, "noise_mse_all_flights_gt", lower_is_better=True)
        best_clustered, worst_clustered = _best_and_worst(
            valid_noise, "noise_mse_clustered_gt", lower_is_better=True
        )

        row.update(
            {
                "n_experiments": int(len(group)),
                "n_noise_results": int(len(valid_noise)),
                "n_missing_noise_results": int((group["noise_results_status"] != "ok").sum()),
                "mean_noise_mse_clustered_gt": float(valid_noise["noise_mse_clustered_gt"].mean()),
                "median_noise_mse_clustered_gt": float(valid_noise["noise_mse_clustered_gt"].median()),
                "mean_noise_mse_all_flights_gt": float(valid_noise["noise_mse_all_flights_gt"].mean()),
                "median_noise_mse_all_flights_gt": float(valid_noise["noise_mse_all_flights_gt"].median()),
                "mean_noise_frac": float(group["noise_frac"].mean()),
                "mean_clustered_frac": float(group["clustered_frac"].mean()),
                "mean_silhouette_valid": float(group["silhouette_valid"].mean()),
                "mean_davies_bouldin_valid": float(group["davies_bouldin_valid"].mean()),
                "mean_flows_n_clusters_gt1": float(group["flows_n_clusters_gt1"].mean()),
                "best_exp_all_flights_gt": best_all,
                "worst_exp_all_flights_gt": worst_all,
                "best_exp_clustered_gt": best_clustered,
                "worst_exp_clustered_gt": worst_clustered,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _corr_row(scope_type: str, scope_value: str, x_col: str, y_col: str, frame: pd.DataFrame) -> dict[str, object]:
    subset = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 3 or subset[x_col].nunique() < 2 or subset[y_col].nunique() < 2:
        return {
            "scope_type": scope_type,
            "scope_value": scope_value,
            "x_metric": x_col,
            "y_metric": y_col,
            "n": int(len(subset)),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
        }

    pearson_r, pearson_p = stats.pearsonr(subset[x_col], subset[y_col])
    spearman_rho, spearman_p = stats.spearmanr(subset[x_col], subset[y_col])
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "x_metric": x_col,
        "y_metric": y_col,
        "n": int(len(subset)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def _build_correlations(experiments_df: pd.DataFrame) -> pd.DataFrame:
    target_cols = ["noise_mse_clustered_gt", "noise_mse_all_flights_gt"]
    feature_cols = [
        "silhouette_valid",
        "davies_bouldin_valid",
        "noise_frac",
        "clustered_frac",
        "flows_n_clusters_gt1",
        "n_clusters_total",
    ]
    valid = experiments_df[experiments_df["noise_results_status"].eq("ok")].copy()
    rows: list[dict[str, object]] = []

    for target in target_cols:
        for feature in feature_cols:
            rows.append(_corr_row("all", "EXP001-062", feature, target, valid))
        for block, block_df in valid.groupby("experiment_block"):
            for feature in feature_cols:
                rows.append(_corr_row("experiment_block", str(block), feature, target, block_df))

    return pd.DataFrame(rows)


def _fmt(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _build_report(
    experiments_df: pd.DataFrame,
    block_df: pd.DataFrame,
    method_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> str:
    valid = experiments_df[experiments_df["noise_results_status"].eq("ok")].copy()
    valid_all = valid.dropna(subset=["noise_mse_all_flights_gt"]).sort_values("noise_mse_all_flights_gt")
    valid_clustered = valid.dropna(subset=["noise_mse_clustered_gt"]).sort_values("noise_mse_clustered_gt")

    best_all = valid_all.iloc[0] if not valid_all.empty else None
    worst_all = valid_all.iloc[-1] if not valid_all.empty else None
    best_clustered = valid_clustered.iloc[0] if not valid_clustered.empty else None

    valid_blocks = block_df[
        block_df["n_noise_results"].fillna(0).gt(0) & block_df["mean_noise_mse_all_flights_gt"].notna()
    ].copy()
    best_block = valid_blocks.sort_values("mean_noise_mse_all_flights_gt").iloc[0] if not valid_blocks.empty else None
    worst_block = valid_blocks.sort_values("mean_noise_mse_all_flights_gt").iloc[-1] if not valid_blocks.empty else None

    corr_target = corr_df[
        (corr_df["scope_type"] == "all")
        & (corr_df["scope_value"] == "EXP001-062")
        & (corr_df["y_metric"] == "noise_mse_all_flights_gt")
        & (corr_df["spearman_rho"].notna())
    ].copy()
    corr_target["abs_spearman"] = corr_target["spearman_rho"].abs()
    strongest_corr = corr_target.sort_values("abs_spearman", ascending=False).iloc[0] if not corr_target.empty else None

    method_view = method_df.sort_values(["mean_noise_mse_all_flights_gt", "method", "distance_metric"])
    top_methods = method_view.head(5)

    missing_noise = experiments_df[experiments_df["noise_results_status"] != "ok"]["exp"].tolist()

    lines = [
        "# Noise and Clustering Results: EXP001-EXP062",
        "",
        "## Scope",
        "This note links clustering quality and Doc29 noise-simulation performance for experiments `EXP001..EXP062`.",
        "",
        "Primary evaluation axis:",
        "- `noise_mse_all_flights_gt`: thesis-level metric; compares experiment prediction against the shared all-flights ground truth and therefore penalizes both fit error and coverage loss.",
        "- `noise_mse_clustered_gt`: conditional fit metric; compares clustered prediction against clustered-category ground truth only.",
        "",
        "Generated from:",
        "- `output/eda/exp001_062_noise_clustering_experiment_summary.csv`",
        "- `output/eda/exp001_062_noise_clustering_block_summary.csv`",
        "- `output/eda/exp001_062_noise_clustering_method_summary.csv`",
        "- `output/eda/exp001_062_noise_clustering_correlations.csv`",
        "",
        "## Headline Findings",
    ]

    if best_all is not None:
        lines.append(
            f"- Best overall experiment by all-flights noise MSE: **{best_all['exp']}** "
            f"({best_all['method']}, {best_all['distance_metric']}, preprocessed_{best_all['preprocessed_id']}; "
            f"`MSE_all={_fmt(best_all['noise_mse_all_flights_gt'])}`, "
            f"`noise_frac={_fmt(best_all['noise_frac'])}`, "
            f"`silhouette_valid={_fmt(best_all['silhouette_valid'])}`)."
        )
    if worst_all is not None:
        lines.append(
            f"- Worst overall experiment by all-flights noise MSE: **{worst_all['exp']}** "
            f"({worst_all['method']}, {worst_all['distance_metric']}, preprocessed_{worst_all['preprocessed_id']}; "
            f"`MSE_all={_fmt(worst_all['noise_mse_all_flights_gt'])}`, "
            f"`noise_frac={_fmt(worst_all['noise_frac'])}`)."
        )
    if best_clustered is not None:
        lines.append(
            f"- Best clustered-only Doc29 fit: **{best_clustered['exp']}** "
            f"with `MSE_clustered={_fmt(best_clustered['noise_mse_clustered_gt'])}`."
        )
    if best_block is not None and worst_block is not None:
        lines.append(
            f"- Best block by mean all-flights noise MSE: **{best_block['experiment_block']} {best_block['block_label']}** "
            f"(`mean={_fmt(best_block['mean_noise_mse_all_flights_gt'])}`)."
        )
        lines.append(
            f"- Weakest block by mean all-flights noise MSE: **{worst_block['experiment_block']} {worst_block['block_label']}** "
            f"(`mean={_fmt(worst_block['mean_noise_mse_all_flights_gt'])}`)."
        )
    if strongest_corr is not None:
        lines.append(
            f"- Strongest global monotonic relationship with `noise_mse_all_flights_gt`: "
            f"`{strongest_corr['x_metric']}` with Spearman `rho={_fmt(strongest_corr['spearman_rho'])}` "
            f"(p=`{_fmt(strongest_corr['spearman_p'])}`)."
        )

    lines.extend(
        [
            "",
            "## Block Summary",
            "| Block | Label | n | mean MSE all-flights | mean MSE clustered | mean noise frac | mean silhouette_valid |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in block_df.sort_values("experiment_block").iterrows():
        lines.append(
            f"| {row['experiment_block']} | {row['block_label']} | {_fmt(row['n_experiments'], 0)} | "
            f"{_fmt(row['mean_noise_mse_all_flights_gt'])} | {_fmt(row['mean_noise_mse_clustered_gt'])} | "
            f"{_fmt(row['mean_noise_frac'])} | {_fmt(row['mean_silhouette_valid'])} |"
        )

    lines.extend(
        [
            "",
            "## Method and Distance Summary",
            "| Method | Distance | n | mean MSE all-flights | mean noise frac | mean clustered frac |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_methods.iterrows():
        lines.append(
            f"| {row['method']} | {row['distance_metric']} | {_fmt(row['n_experiments'], 0)} | "
            f"{_fmt(row['mean_noise_mse_all_flights_gt'])} | {_fmt(row['mean_noise_frac'])} | "
            f"{_fmt(row['mean_clustered_frac'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- Use the experiment-level CSV when you need exact case comparisons in the thesis text.",
            "- Use the block summary when discussing research-question trends or method families.",
            "- Prefer `noise_mse_all_flights_gt` for the final overall ranking, because it exposes cases where apparent cluster quality comes from labeling large parts of the dataset as noise or from evaluating only a reduced subset.",
        ]
    )

    if missing_noise:
        lines.extend(
            [
                "",
                "## Missing Noise Outputs",
                f"- Missing or incomplete Doc29 aggregate outputs were detected for: {', '.join(missing_noise)}.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize noise-simulation and clustering results for EXP001-EXP062.")
    parser.add_argument("--start", type=int, default=1, help="Start experiment number (default: 1).")
    parser.add_argument("--end", type=int, default=62, help="End experiment number (default: 62).")
    parser.add_argument("--experiments-root", default="output/experiments", help="Experiment output root.")
    parser.add_argument("--noise-root", default="noise_simulation/results", help="Noise results root.")
    parser.add_argument("--eda-dir", default="output/eda", help="EDA output directory.")
    parser.add_argument(
        "--report-path",
        default="thesis/docs/noise_clustering_results_exp001_062.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    experiments_root = (REPO_ROOT / args.experiments_root).resolve()
    noise_root = (REPO_ROOT / args.noise_root).resolve()
    eda_dir = (REPO_ROOT / args.eda_dir).resolve()
    report_path = (REPO_ROOT / args.report_path).resolve()

    eda_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    experiment_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    for exp_num in range(args.start, args.end + 1):
        exp = _exp_name(exp_num)
        exp_dir = experiments_root / exp
        noise_dir = noise_root / exp
        block_code, block_label = _exp_block(exp_num)

        config = _read_yaml(exp_dir / "config_resolved.yaml")
        metrics = _scalar_from_csv(exp_dir / "metrics_global.csv")
        labels = _labels_summary(exp_dir / "labels_ALL.csv")
        overall_noise = _read_json(noise_dir / "aggregate_totals" / "overall_summary.json")
        flow_summary = _noise_flow_summary(
            noise_dir / "summary_mse.csv",
            noise_dir / "aggregate_totals" / "category_summary.csv",
        )
        noise_status = _status_for_noise_results(noise_dir)

        clustering_cfg = config.get("clustering", {})
        input_cfg = config.get("input", {})
        features_cfg = config.get("features", {})
        vector_cols = features_cfg.get("vector_cols") or []

        row = {
            "exp": exp,
            "exp_num": exp_num,
            "experiment_block": block_code,
            "block_label": block_label,
            "method": clustering_cfg.get("method"),
            "distance_metric": clustering_cfg.get("distance_metric"),
            "preprocessed_csv": input_cfg.get("preprocessed_csv"),
            "preprocessed_id": _parse_preprocessed_id(input_cfg.get("preprocessed_csv")),
            "vector_cols": ",".join(vector_cols),
            "uses_altitude": "altitude" in set(vector_cols),
            "sample_for_fit_enabled": bool(clustering_cfg.get("sample_for_fit", {}).get("enabled", False)),
            "noise_results_status": noise_status,
            "n_categories": overall_noise.get("n_categories"),
            "skipped_rows": overall_noise.get("skipped_rows"),
            "silhouette": metrics.get("silhouette"),
            "davies_bouldin": metrics.get("davies_bouldin"),
            "calinski_harabasz": metrics.get("calinski_harabasz"),
            "n_flows_total": metrics.get("n_flows_total"),
            "n_flows_valid": metrics.get("n_flows_valid"),
            "silhouette_valid": metrics.get("silhouette_valid"),
            "davies_bouldin_valid": metrics.get("davies_bouldin_valid"),
            "calinski_harabasz_valid": metrics.get("calinski_harabasz_valid"),
            "total_flights": labels.get("total_flights", metrics.get("total_flights")),
            "total_noise_flights": labels.get("total_noise_flights", metrics.get("total_noise_flights")),
            "total_clustered_flights": labels.get("total_clustered_flights", metrics.get("total_clustered_flights")),
            "noise_frac": labels.get("noise_frac", metrics.get("noise_frac")),
            "clustered_frac": labels.get("clustered_frac"),
            "n_flows_total_labels": labels.get("n_flows_total_labels"),
            "flows_n_clusters_gt1": labels.get("flows_n_clusters_gt1"),
            "n_clusters_total": labels.get("n_clusters_total"),
            "noise_mae_clustered_gt": overall_noise.get("mae_cumulative_res"),
            "noise_mse_clustered_gt": overall_noise.get("mse_cumulative_res"),
            "noise_rmse_clustered_gt": overall_noise.get("rmse_cumulative_res"),
            "noise_delta_avg_clustered_gt": overall_noise.get("delta_avg_cumulative_res"),
            "noise_mae_all_flights_gt": overall_noise.get("mae_cumulative_res_all_flights"),
            "noise_mse_all_flights_gt": overall_noise.get("mse_cumulative_res_all_flights"),
            "noise_rmse_all_flights_gt": overall_noise.get("rmse_cumulative_res_all_flights"),
            "noise_delta_avg_all_flights_gt": overall_noise.get("delta_avg_cumulative_res_all_flights"),
        }
        row["noise_mse_gap_all_minus_clustered"] = (
            _float_or_nan(row["noise_mse_all_flights_gt"]) - _float_or_nan(row["noise_mse_clustered_gt"])
        )

        if not flow_summary.empty:
            worst_flow = flow_summary.sort_values("weighted_mse", ascending=False).iloc[0]
            best_flow = flow_summary.sort_values("weighted_mse", ascending=True).iloc[0]
            row["worst_flow_label"] = worst_flow["flow_label"]
            row["worst_flow_weighted_mse"] = float(worst_flow["weighted_mse"])
            row["best_flow_label"] = best_flow["flow_label"]
            row["best_flow_weighted_mse"] = float(best_flow["weighted_mse"])

            flow_summary = flow_summary.copy()
            flow_summary.insert(0, "exp", exp)
            flow_summary.insert(1, "exp_num", exp_num)
            flow_summary.insert(2, "experiment_block", block_code)
            flow_rows.extend(flow_summary.to_dict(orient="records"))
        else:
            row["worst_flow_label"] = None
            row["worst_flow_weighted_mse"] = float("nan")
            row["best_flow_label"] = None
            row["best_flow_weighted_mse"] = float("nan")

        experiment_rows.append(row)

    experiments_df = pd.DataFrame(experiment_rows).sort_values("exp_num").reset_index(drop=True)
    valid_noise = experiments_df[experiments_df["noise_results_status"].eq("ok")].copy()

    if not valid_noise.empty:
        experiments_df["rank_noise_mse_all_flights"] = valid_noise["noise_mse_all_flights_gt"].rank(method="min")
        experiments_df["rank_noise_mse_clustered"] = valid_noise["noise_mse_clustered_gt"].rank(method="min")
        experiments_df["rank_silhouette_valid_desc"] = (
            valid_noise["silhouette_valid"].rank(method="min", ascending=False)
        )
    else:
        experiments_df["rank_noise_mse_all_flights"] = float("nan")
        experiments_df["rank_noise_mse_clustered"] = float("nan")
        experiments_df["rank_silhouette_valid_desc"] = float("nan")

    flow_df = pd.DataFrame(flow_rows)
    if not flow_df.empty:
        flow_df = flow_df.sort_values(["exp_num", "A/D", "Runway"]).reset_index(drop=True)
    block_df = _group_summary(experiments_df, ["experiment_block", "block_label"]).sort_values("experiment_block")
    method_df = _group_summary(experiments_df, ["method", "distance_metric"]).sort_values(
        ["mean_noise_mse_all_flights_gt", "method", "distance_metric"],
        na_position="last",
    )
    corr_df = _build_correlations(experiments_df).sort_values(
        ["scope_type", "scope_value", "y_metric", "x_metric"]
    )

    experiment_csv = eda_dir / "exp001_062_noise_clustering_experiment_summary.csv"
    flow_csv = eda_dir / "exp001_062_noise_clustering_flow_summary.csv"
    block_csv = eda_dir / "exp001_062_noise_clustering_block_summary.csv"
    method_csv = eda_dir / "exp001_062_noise_clustering_method_summary.csv"
    corr_csv = eda_dir / "exp001_062_noise_clustering_correlations.csv"

    experiments_df.to_csv(experiment_csv, index=False)
    flow_df.to_csv(flow_csv, index=False)
    block_df.to_csv(block_csv, index=False)
    method_df.to_csv(method_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)

    report_path.write_text(
        _build_report(experiments_df, block_df, method_df, corr_df),
        encoding="utf-8",
    )

    print(f"Wrote {experiment_csv}")
    print(f"Wrote {flow_csv}")
    print(f"Wrote {block_csv}")
    print(f"Wrote {method_csv}")
    print(f"Wrote {corr_csv}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
