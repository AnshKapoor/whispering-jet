"""Export a master verification table for EXP001-EXP050.

Inputs:
  - output/experiments/EXP###/metrics_global.csv
  - output/experiments/EXP###/metrics_by_flow.csv
  - output/experiments/EXP###/cluster_counts_by_flow.csv

Outputs:
  - output/eda/exp001_050_master_results_table.csv
  - thesis/docs/exp001_050_master_results_table.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "output" / "experiments"
OUTPUT_CSV = REPO_ROOT / "output" / "eda" / "exp001_050_master_results_table.csv"
OUTPUT_MD = REPO_ROOT / "thesis" / "docs" / "exp001_050_master_results_table.md"

FLOW_ORDER = [
    ("Landung", "09L", "A09L"),
    ("Landung", "09R", "A09R"),
    ("Landung", "27L", "A27L"),
    ("Landung", "27R", "A27R"),
    ("Start", "09L", "D09L"),
    ("Start", "09R", "D09R"),
    ("Start", "27L", "D27L"),
    ("Start", "27R", "D27R"),
]


@dataclass(frozen=True)
class FlowKey:
    """Flow identifier used to join per-flow metrics and cluster counts."""

    ad: str
    runway: str
    scope: str


def _experiment_name(exp_num: int) -> str:
    return f"EXP{exp_num:03d}"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return pd.read_csv(path)


def _flow_keys() -> list[FlowKey]:
    return [FlowKey(ad=ad, runway=runway, scope=scope) for ad, runway, scope in FLOW_ORDER]


def _safe_float(value: object) -> float | None:
    try:
        if pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(round(parsed))


def _format_metric(value: object, decimals: int = 3) -> str:
    parsed = _safe_float(value)
    return "-" if parsed is None else f"{parsed:.{decimals}f}"


def _format_ch(value: object) -> str:
    parsed = _safe_float(value)
    return "-" if parsed is None else f"{parsed:.0f}"


def _format_count(value: object) -> str:
    parsed = _safe_int(value)
    return "-" if parsed is None else f"{parsed:,}"


def _cluster_signature(cluster_counts: pd.DataFrame) -> str:
    """Return a compact count signature like '0=428 / 1=163 + 2307 noise'."""

    if cluster_counts.empty:
        return "-"

    ordered = cluster_counts.copy()
    ordered["cluster_id"] = pd.to_numeric(ordered["cluster_id"], errors="coerce")
    ordered["n_flights"] = pd.to_numeric(ordered["n_flights"], errors="coerce").fillna(0).astype(int)
    ordered = ordered.sort_values("cluster_id")

    noise_total = int(
        ordered.loc[ordered["cluster_id"].eq(-1), "n_flights"].sum()
    )
    cluster_parts = [
        f"{int(cluster_id)}={int(n_flights)}"
        for cluster_id, n_flights in ordered.loc[ordered["cluster_id"].ne(-1), ["cluster_id", "n_flights"]].itertuples(index=False)
    ]
    if not cluster_parts:
        return f"noise={noise_total}"
    if noise_total > 0:
        return f"{' / '.join(cluster_parts)} + {noise_total} noise"
    return " / ".join(cluster_parts)


def _overall_flow_signature(rows: Iterable[dict[str, object]]) -> str:
    parts: list[str] = []
    for row in rows:
        parts.append(f"{row['Scope']}={row['k']}")
    return "; ".join(parts)


def _build_table_rows(exp_num: int) -> list[dict[str, object]]:
    exp_name = _experiment_name(exp_num)
    exp_dir = EXPERIMENTS_DIR / exp_name

    metrics_global = _read_csv(exp_dir / "metrics_global.csv")
    metrics_by_flow = _read_csv(exp_dir / "metrics_by_flow.csv")
    cluster_counts = _read_csv(exp_dir / "cluster_counts_by_flow.csv")

    metrics_global_row = metrics_global.iloc[0]

    flow_rows: list[dict[str, object]] = []
    for flow_key in _flow_keys():
        flow_metric = metrics_by_flow[
            (metrics_by_flow["A/D"] == flow_key.ad) & (metrics_by_flow["Runway"].astype(str) == flow_key.runway)
        ]
        if flow_metric.empty:
            raise ValueError(f"Missing flow metrics for {exp_name} {flow_key.scope}")

        flow_metric_row = flow_metric.iloc[0]
        flow_clusters = cluster_counts[
            (cluster_counts["A/D"] == flow_key.ad) & (cluster_counts["Runway"].astype(str) == flow_key.runway)
        ]

        flow_rows.append(
            {
                "Experiment": exp_name,
                "Scope": flow_key.scope,
                "N": _safe_int(flow_metric_row["n_flights_total_flow"]),
                "k": _safe_int(flow_metric_row["n_clusters"]),
                "NoisePct": _safe_float(flow_metric_row["noise_frac"]) * 100.0
                if _safe_float(flow_metric_row["noise_frac"]) is not None
                else None,
                "Silhouette": _safe_float(flow_metric_row["silhouette"]),
                "DaviesBouldin": _safe_float(flow_metric_row["davies_bouldin"]),
                "CalinskiHarabasz": _safe_float(flow_metric_row["calinski_harabasz"]),
                "ClusterSignature": _cluster_signature(flow_clusters),
            }
        )

    total_clusters = sum(int(row["k"]) for row in flow_rows if row["k"] is not None)
    overall_row = {
        "Experiment": exp_name,
        "Scope": "Overall",
        "N": _safe_int(metrics_global_row["total_flights"]),
        "k": total_clusters,
        "NoisePct": _safe_float(metrics_global_row["noise_frac"]) * 100.0
        if _safe_float(metrics_global_row["noise_frac"]) is not None
        else None,
        "Silhouette": _safe_float(metrics_global_row["silhouette"]),
        "DaviesBouldin": _safe_float(metrics_global_row["davies_bouldin"]),
        "CalinskiHarabasz": _safe_float(metrics_global_row["calinski_harabasz"]),
        "ClusterSignature": _overall_flow_signature(flow_rows),
    }

    return [overall_row, *flow_rows]


def _build_master_table(start_exp: int = 1, end_exp: int = 50) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for exp_num in range(start_exp, end_exp + 1):
        rows.extend(_build_table_rows(exp_num))
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    header = [
        "| Exp | Scope | N | k | Noise % | Sil. | DB | CH | Cluster signature |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    body: list[str] = []
    for row in df.itertuples(index=False):
        body.append(
            "| {exp} | {scope} | {n} | {k} | {noise} | {sil} | {db} | {ch} | {sig} |".format(
                exp=row.Experiment,
                scope=row.Scope,
                n=_format_count(row.N),
                k=_format_count(row.k),
                noise="-" if row.NoisePct is None else f"{row.NoisePct:.2f}",
                sil=_format_metric(row.Silhouette),
                db=_format_metric(row.DaviesBouldin),
                ch=_format_ch(row.CalinskiHarabasz),
                sig=row.ClusterSignature,
            )
        )
    return "\n".join(header + body)


def main() -> None:
    """Build and write the master verification table in CSV and Markdown."""

    df = _build_master_table()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    markdown = "\n".join(
        [
            "# EXP001-EXP050 Master Results Table",
            "",
            "Canonical verification table for the clustering results discussion.",
            "",
            "Notes:",
            "- `Scope = Overall` uses the experiment-level metrics from `metrics_global.csv`.",
            "- `Scope = A09L/A09R/A27L/A27R/D09L/D09R/D27L/D27R` uses per-flow metrics from `metrics_by_flow.csv`.",
            "- `k` is the number of non-noise clusters.",
            "- `Noise %` is the fraction of noise flights in percent.",
            "- `Sil.`, `DB`, and `CH` are left blank when the flow has fewer than 2 non-noise clusters.",
            "- `Cluster signature` lists non-noise cluster sizes and the noise count.",
            "",
            _markdown_table(df),
            "",
        ]
    )
    OUTPUT_MD.write_text(markdown, encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
