"""Write brief batch markdown summaries for noise-simulation experiments.

Inputs consumed per experiment:
- output/experiments/EXPxxx/config_resolved.yaml
- output/experiments/EXPxxx/metrics_global.csv
- output/experiments/EXPxxx/metrics_by_flow.csv
- noise_simulation/results/EXPxxx/aggregate_totals/overall_summary.json
- noise_simulation/results/EXPxxx/aggregate_totals/category_summary.csv
- noise_simulation/results/EXPxxx/aggregate_totals/category_aligned_receivers.csv
- noise_simulation/results/EXPxxx/aggregate_totals/overall_aligned_9points.csv

Outputs written:
- noise_simulation/generated_markdowns/exp001_062/00_index.md
- one markdown file per logical batch between EXP001 and EXP062

The summaries are intentionally brief and thesis-oriented:
- experiment table with clustered-vs-all-flights ground truth metrics
- short batch takeaways
- a small table of hardest categories
- a small table of worst receiver misses versus all-flights ground truth
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "noise_simulation" / "generated_markdowns" / "exp001_062"


@dataclass(frozen=True)
class BatchDef:
    """Logical batch metadata for grouped markdown outputs."""

    slug: str
    title: str
    notes: str
    experiments: Sequence[int]


BATCHES: List[BatchDef] = [
    BatchDef(
        slug="01_exp001_010_optics_preprocessing_sweep",
        title="EXP001-EXP010: OPTICS preprocessing sweep",
        notes="Same OPTICS settings, different preprocessed inputs.",
        experiments=tuple(range(1, 11)),
    ),
    BatchDef(
        slug="02_exp011_020_kmeans_hdbscan_baselines",
        title="EXP011-EXP020: KMeans and HDBSCAN baselines",
        notes="Vector-space baselines on Euclidean features with preprocessing sensitivity.",
        experiments=tuple(range(11, 21)),
    ),
    BatchDef(
        slug="03_exp021_024_dtw_frechet_pilot",
        title="EXP021-EXP024: DTW and Frechet pilot runs",
        notes="First non-Euclidean distance block using HDBSCAN and OPTICS.",
        experiments=tuple(range(21, 25)),
    ),
    BatchDef(
        slug="04_exp025_031_optics_parameter_sweep",
        title="EXP025-EXP031: OPTICS parameter sweep",
        notes="Euclidean OPTICS sensitivity around min_samples, xi, and min_cluster_size.",
        experiments=tuple(range(25, 32)),
    ),
    BatchDef(
        slug="05_exp032_036_dbscan_gap",
        title="EXP032-EXP036: DBSCAN block",
        notes="Noise-simulation outputs are incomplete or missing in this block.",
        experiments=tuple(range(32, 37)),
    ),
    BatchDef(
        slug="06_exp037_046_gmm_agglomerative_birch",
        title="EXP037-EXP046: GMM, agglomerative, and Birch baselines",
        notes="Classical baseline families on Euclidean trajectory vectors.",
        experiments=tuple(range(37, 47)),
    ),
    BatchDef(
        slug="07_exp047_050_preprocessed3_cross_algorithm",
        title="EXP047-EXP050: Cross-algorithm reruns on preprocessed_3",
        notes="Same preprocessed input, algorithm family changes.",
        experiments=tuple(range(47, 51)),
    ),
    BatchDef(
        slug="08_exp051_057_dtw_reruns",
        title="EXP051-EXP057: DTW reruns",
        notes="Post-50 DTW-focused runs across HDBSCAN and OPTICS.",
        experiments=tuple(range(51, 58)),
    ),
    BatchDef(
        slug="09_exp058_062_lcss_sample_only",
        title="EXP058-EXP062: LCSS sample-only runs",
        notes="LCSS runs with `sample_only` fitting and 1200-flight caps per flow.",
        experiments=tuple(range(58, 63)),
    ),
]


def exp_name(num: int) -> str:
    """Return EXP-style zero-padded experiment name."""

    return f"EXP{num:03d}"


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML file into a dict."""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dict."""

    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: Any, digits: int = 2) -> str:
    """Format floats for markdown tables with NA handling."""

    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number) or math.isinf(number):
        return "NA"
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    """Format proportions as percentages."""

    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number) or math.isinf(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def short_prep(name: str | None) -> str:
    """Shorten preprocessed ids for compact tables."""

    if not name:
        return "NA"
    return name.replace("preprocessed_", "prep")


def concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate only non-empty frames and return an empty frame otherwise."""

    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True)


def setup_label(record: Dict[str, Any]) -> str:
    """Build a compact setup label for a single experiment."""

    parts = [
        record.get("method", "NA"),
        record.get("distance_metric", "NA"),
        short_prep(record.get("preprocessed_id")),
    ]
    if record.get("params"):
        parts.append(str(record["params"]))
    if record.get("sample_for_fit_mode"):
        max_fit = record.get("sample_for_fit_max")
        try:
            max_fit_value = float(max_fit)
        except (TypeError, ValueError):
            max_fit_value = math.nan
        if not math.isnan(max_fit_value):
            parts.append(f"{record['sample_for_fit_mode']}:{int(max_fit_value)}")
        else:
            parts.append(str(record["sample_for_fit_mode"]))
    return ", ".join(parts)


def make_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a plain markdown table without third-party dependencies."""

    safe_rows = []
    for row in rows:
        safe_rows.append(
            [str(cell).replace("\n", " ").replace("|", "/") if cell is not None else "" for cell in row]
        )
    header_line = "| " + " | ".join(headers) + " |"
    divider_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in safe_rows]
    return "\n".join([header_line, divider_line, *body_lines])


def weighted_noise_fraction(metrics_by_flow: pd.DataFrame) -> float | None:
    """Compute a flight-weighted noise fraction from per-flow metrics."""

    if metrics_by_flow.empty or "noise_frac" not in metrics_by_flow.columns:
        return None
    if "n_flights" not in metrics_by_flow.columns:
        return None
    weights = pd.to_numeric(metrics_by_flow["n_flights"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(metrics_by_flow["noise_frac"], errors="coerce").fillna(0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return None
    return float((weights * values).sum() / total_weight)


def load_experiment_record(exp_num: int) -> Dict[str, Any]:
    """Load metadata and aggregate outputs for one experiment."""

    exp = exp_name(exp_num)
    output_dir = REPO_ROOT / "output" / "experiments" / exp
    noise_dir = REPO_ROOT / "noise_simulation" / "results" / exp
    agg_dir = noise_dir / "aggregate_totals"
    config_path = output_dir / "config_resolved.yaml"
    metrics_global_path = output_dir / "metrics_global.csv"
    metrics_by_flow_path = output_dir / "metrics_by_flow.csv"
    overall_summary_path = agg_dir / "overall_summary.json"
    category_summary_path = agg_dir / "category_summary.csv"
    category_aligned_path = agg_dir / "category_aligned_receivers.csv"
    overall_aligned_path = agg_dir / "overall_aligned_9points.csv"

    record: Dict[str, Any] = {
        "experiment": exp,
        "status": "missing",
        "config_path": config_path,
        "noise_dir": noise_dir,
        "category_rows": pd.DataFrame(),
        "category_hotspots": pd.DataFrame(),
        "overall_receivers": pd.DataFrame(),
    }

    if config_path.exists():
        config = read_yaml(config_path)
        clustering = config.get("clustering", {})
        features = config.get("features", {})
        record.update(
            {
                "status": "config_only",
                "method": clustering.get("method"),
                "distance_metric": clustering.get("distance_metric"),
                "preprocessed_id": Path(
                    str(config.get("input", {}).get("preprocessed_csv", ""))
                ).stem
                or None,
                "vector_cols": ",".join(features.get("vector_cols", []) or []),
            }
        )
        method = record.get("method")
        if method == "optics":
            optics = clustering.get("optics", {})
            record["params"] = (
                f"ms={optics.get('min_samples')},"
                f"xi={optics.get('xi')},"
                f"mcs={optics.get('min_cluster_size')}"
            )
        elif method == "hdbscan":
            hdbscan = clustering.get("hdbscan", {})
            record["params"] = (
                f"mcs={hdbscan.get('min_cluster_size')},"
                f"ms={hdbscan.get('min_samples')}"
            )
        elif method == "kmeans":
            kmeans = clustering.get("kmeans", {})
            record["params"] = f"k={kmeans.get('n_clusters')}"
        elif method == "minibatch_kmeans":
            kmeans = clustering.get("minibatch_kmeans", {})
            record["params"] = f"k={kmeans.get('n_clusters')}"
        elif method == "dbscan":
            dbscan = clustering.get("dbscan", {})
            record["params"] = f"eps={dbscan.get('eps')},ms={dbscan.get('min_samples')}"
        elif method == "agglomerative":
            agglomerative = clustering.get("agglomerative", {})
            record["params"] = (
                f"k={agglomerative.get('n_clusters')},"
                f"link={agglomerative.get('linkage')}"
            )
        elif method == "birch":
            birch = clustering.get("birch", {})
            record["params"] = f"k={birch.get('n_clusters')}"
        elif method == "gmm":
            gmm = clustering.get("gmm", {})
            record["params"] = (
                f"k={gmm.get('n_components')},"
                f"cov={gmm.get('covariance_type')}"
            )

        sample_for_fit = clustering.get("sample_for_fit") or {}
        if sample_for_fit:
            record["sample_for_fit_mode"] = sample_for_fit.get("mode")
            record["sample_for_fit_max"] = sample_for_fit.get("max_flights_per_flow")

    metrics_by_flow = pd.read_csv(metrics_by_flow_path) if metrics_by_flow_path.exists() else pd.DataFrame()
    metrics_global = pd.read_csv(metrics_global_path) if metrics_global_path.exists() else pd.DataFrame()
    if not metrics_by_flow.empty:
        record["noise_frac"] = weighted_noise_fraction(metrics_by_flow)
        if "n_clusters" in metrics_by_flow.columns:
            valid = pd.to_numeric(metrics_by_flow["n_clusters"], errors="coerce").fillna(0)
            record["n_flows_valid"] = int((valid >= 2).sum())
        if "n_flights" in metrics_by_flow.columns:
            total_flights = pd.to_numeric(metrics_by_flow["n_flights"], errors="coerce").fillna(0).sum()
            record["n_flights_total"] = int(total_flights)
    if not metrics_global.empty:
        global_row = metrics_global.iloc[0].to_dict()
        if record.get("noise_frac") is None and "noise_frac" in global_row:
            record["noise_frac"] = global_row.get("noise_frac")
        if record.get("n_flows_valid") is None and "n_flows_valid" in global_row:
            try:
                record["n_flows_valid"] = int(float(global_row["n_flows_valid"]))
            except (TypeError, ValueError):
                pass

    if overall_summary_path.exists():
        overall_summary = read_json(overall_summary_path)
        record.update(
            {
                "status": "available",
                "rmse_clustered": overall_summary.get("rmse_cumulative_res"),
                "rmse_all_flights": overall_summary.get("rmse_cumulative_res_all_flights"),
                "mae_clustered": overall_summary.get("mae_cumulative_res"),
                "mae_all_flights": overall_summary.get("mae_cumulative_res_all_flights"),
                "delta_clustered": overall_summary.get("delta_avg_cumulative_res"),
                "delta_all_flights": overall_summary.get("delta_avg_cumulative_res_all_flights"),
                "n_categories": overall_summary.get("n_categories"),
                "global_ground_truth_csv": overall_summary.get("global_ground_truth_csv"),
            }
        )

        if category_summary_path.exists():
            category_df = pd.read_csv(category_summary_path)
            if not category_df.empty:
                category_df = category_df.copy()
                category_df["experiment"] = exp
                category_df["category_label"] = (
                    category_df["A/D"].astype(str)
                    + " / "
                    + category_df["Runway"].astype(str)
                    + " / "
                    + category_df["aircraft_type"].astype(str)
                )
                record["category_rows"] = category_df
                record["category_rmse_mean"] = float(category_df["rmse_cumulative_res"].mean())
                record["category_rmse_p90"] = float(category_df["rmse_cumulative_res"].quantile(0.90))
                worst_idx = category_df["rmse_cumulative_res"].idxmax()
                worst_row = category_df.loc[worst_idx]
                record["worst_category_label"] = worst_row["category_label"]
                record["worst_category_rmse"] = float(worst_row["rmse_cumulative_res"])

        if category_aligned_path.exists():
            aligned_df = pd.read_csv(category_aligned_path)
            if not aligned_df.empty:
                aligned_df = aligned_df.copy()
                aligned_df["experiment"] = exp
                aligned_df["category_label"] = (
                    aligned_df["A/D"].astype(str)
                    + " / "
                    + aligned_df["Runway"].astype(str)
                    + " / "
                    + aligned_df["aircraft_type"].astype(str)
                )
                mean_abs = (
                    aligned_df.groupby("measuring_point", dropna=False)["abs_err"]
                    .mean()
                    .sort_values(ascending=False)
                )
                if not mean_abs.empty:
                    record["category_hotspot_receiver"] = str(mean_abs.index[0])
                    record["category_hotspot_abs_err"] = float(mean_abs.iloc[0])
                hotspot_idx = aligned_df.groupby("category_label")["abs_err"].idxmax()
                record["category_hotspots"] = aligned_df.loc[
                    hotspot_idx,
                    ["experiment", "category_label", "measuring_point", "abs_err"],
                ].reset_index(drop=True)

        if overall_aligned_path.exists():
            overall_df = pd.read_csv(overall_aligned_path)
            if not overall_df.empty:
                overall_df = overall_df.copy()
                overall_df["experiment"] = exp
                record["overall_receivers"] = overall_df
                if "abs_err" in overall_df.columns:
                    clustered_idx = overall_df["abs_err"].idxmax()
                    clustered_row = overall_df.loc[clustered_idx]
                    record["worst_receiver_clustered"] = clustered_row["measuring_point"]
                    record["worst_receiver_clustered_abs"] = float(clustered_row["abs_err"])
                if "abs_err_all_flights" in overall_df.columns:
                    all_idx = overall_df["abs_err_all_flights"].idxmax()
                    all_row = overall_df.loc[all_idx]
                    record["worst_receiver_all"] = all_row["measuring_point"]
                    record["worst_receiver_all_abs"] = float(all_row["abs_err_all_flights"])

    elif noise_dir.exists():
        record["status"] = "partial"
        if (noise_dir / "summary_mse.csv").exists():
            record["status_note"] = "summary_mse.csv exists, aggregate_totals missing"
        elif (noise_dir / "run.log").exists():
            record["status_note"] = "run.log stub only"
    else:
        record["status"] = "missing"
        record["status_note"] = "no noise_simulation/results directory"

    return record


def write_batch_markdown(batch: BatchDef, records: Sequence[Dict[str, Any]]) -> None:
    """Write one markdown file for a logical batch."""

    path = OUTPUT_DIR / f"{batch.slug}.md"
    available = [r for r in records if r.get("status") == "available"]
    unavailable = [r for r in records if r.get("status") != "available"]

    lines: List[str] = [
        f"# {batch.title}",
        "",
        batch.notes,
        "",
        "Clustered GT = ground truth summed only over clustered flights.",
        "All-flights GT = global ground truth summed over every flight in the matching preprocessed set.",
        "",
    ]

    if available:
        exp_rows = []
        for record in available:
            exp_rows.append(
                [
                    record["experiment"],
                    setup_label(record),
                    record.get("n_flows_valid", "NA"),
                    fmt_pct(record.get("noise_frac")),
                    fmt_float(record.get("rmse_clustered")),
                    fmt_float(record.get("rmse_all_flights")),
                    record.get("worst_category_label", "NA"),
                ]
            )
        lines.extend(
            [
                "## Experiment table",
                "",
                make_markdown_table(
                    [
                        "Exp",
                        "Setup",
                        "Valid flows",
                        "Noise %",
                        "RMSE clustered",
                        "RMSE all flights",
                        "Worst category",
                    ],
                    exp_rows,
                ),
                "",
            ]
        )

        best_clustered = min(available, key=lambda r: float(r.get("rmse_clustered", math.inf)))
        best_all = min(available, key=lambda r: float(r.get("rmse_all_flights", math.inf)))
        highest_noise = max(available, key=lambda r: float(r.get("noise_frac", -1.0)))
        largest_gap = max(
            available,
            key=lambda r: float(r.get("rmse_all_flights", math.nan))
            - float(r.get("rmse_clustered", math.nan)),
        )
        gap_values = [
            float(r["rmse_all_flights"]) - float(r["rmse_clustered"])
            for r in available
            if r.get("rmse_all_flights") is not None and r.get("rmse_clustered") is not None
        ]

        all_category_rows = concat_frames([r["category_rows"] for r in available])
        all_category_hotspots = concat_frames([r["category_hotspots"] for r in available])
        all_overall_receivers = concat_frames([r["overall_receivers"] for r in available])

        lines.append("## Takeaways")
        lines.append("")
        lines.append(
            f"- Best versus clustered GT: `{best_clustered['experiment']}` with RMSE `{fmt_float(best_clustered.get('rmse_clustered'))}` dB."
        )
        lines.append(
            f"- Best versus all-flights GT: `{best_all['experiment']}` with RMSE `{fmt_float(best_all.get('rmse_all_flights'))}` dB."
        )
        if gap_values:
            largest_gap_value = float(largest_gap["rmse_all_flights"]) - float(largest_gap["rmse_clustered"])
            lines.append(
                f"- RMSE gap (`all-flights - clustered`) averages `{fmt_float(sum(gap_values) / len(gap_values))}` dB; the highest gap is `{largest_gap['experiment']}` at `{fmt_float(largest_gap_value)}` dB."
            )
        lines.append(
            f"- Highest noise share: `{highest_noise['experiment']}` at `{fmt_pct(highest_noise.get('noise_frac'))}`."
        )

        receiver_counts = Counter(
            str(r.get("worst_receiver_all"))
            for r in available
            if r.get("worst_receiver_all")
        )
        if receiver_counts:
            receiver_name, receiver_count = receiver_counts.most_common(1)[0]
            lines.append(
                f"- Receiver hotspot versus all-flights GT: `{receiver_name}` is worst in `{receiver_count}/{len(available)}` available runs."
            )
        lines.append("")

        if not all_category_rows.empty:
            hardest = all_category_rows.sort_values("rmse_cumulative_res", ascending=False).head(3).copy()
            if not all_category_hotspots.empty:
                hardest = hardest.merge(
                    all_category_hotspots.rename(
                        columns={
                            "measuring_point": "hotspot_receiver",
                            "abs_err": "hotspot_abs_err",
                        }
                    ),
                    on=["experiment", "category_label"],
                    how="left",
                )
            lines.extend(["## Hardest categories", ""])
            hardest_rows = []
            for _, row in hardest.iterrows():
                hardest_rows.append(
                    [
                        row["experiment"],
                        row["category_label"],
                        fmt_float(row["rmse_cumulative_res"]),
                        row.get("hotspot_receiver", "NA"),
                        fmt_float(row.get("hotspot_abs_err")),
                    ]
                )
            lines.append(
                make_markdown_table(
                    ["Exp", "Category", "Cat. RMSE", "Hotspot receiver", "Hotspot abs err"],
                    hardest_rows,
                )
            )
            lines.append("")

        if not all_overall_receivers.empty and "abs_err_all_flights" in all_overall_receivers.columns:
            toughest_receivers = (
                all_overall_receivers.sort_values("abs_err_all_flights", ascending=False)
                .head(3)
                .copy()
            )
            lines.extend(["## Worst receiver misses versus all-flights GT", ""])
            receiver_rows = []
            for _, row in toughest_receivers.iterrows():
                receiver_rows.append(
                    [
                        row["experiment"],
                        row["measuring_point"],
                        fmt_float(row["cumulative_res_pred"]),
                        fmt_float(row["cumulative_res_gt_all_flights"]),
                        fmt_float(row["abs_err_all_flights"]),
                    ]
                )
            lines.append(
                make_markdown_table(
                    ["Exp", "Receiver", "Pred", "All-flights GT", "Abs err"],
                    receiver_rows,
                )
            )
            lines.append("")

    if unavailable:
        lines.extend(["## Missing or partial runs", ""])
        status_rows = []
        for record in unavailable:
            status_rows.append(
                [
                    record["experiment"],
                    setup_label(record),
                    record.get("status"),
                    record.get("status_note", ""),
                ]
            )
        lines.append(
            make_markdown_table(
                ["Exp", "Setup", "Status", "Note"],
                status_rows,
            )
        )
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_index(records: Sequence[Dict[str, Any]]) -> None:
    """Write a small index file for all generated batch markdowns."""

    available = sum(1 for r in records if r.get("status") == "available")
    partial = sum(1 for r in records if r.get("status") == "partial")
    missing = sum(1 for r in records if r.get("status") == "missing")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Noise Simulation Batch Summaries: EXP001-EXP062",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"Available aggregate summaries: `{available}`",
        f"Partial runs: `{partial}`",
        f"Missing runs: `{missing}`",
        "",
        "## Files",
        "",
    ]
    for batch in BATCHES:
        lines.append(f"- `{batch.slug}.md`: {batch.title}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `EXP032` is present only as a stub run without `aggregate_totals`.")
    lines.append("- `EXP033-EXP036` have no `noise_simulation/results` directories.")
    lines.append("")

    (OUTPUT_DIR / "00_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate all batch markdown summaries for EXP001-EXP062."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [load_experiment_record(i) for i in range(1, 63)]
    for batch in BATCHES:
        batch_records = [r for r in records if int(r["experiment"][-3:]) in set(batch.experiments)]
        write_batch_markdown(batch, batch_records)
    write_index(records)
    print(f"Wrote markdown summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
