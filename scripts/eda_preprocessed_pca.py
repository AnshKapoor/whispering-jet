"""Compute PCA explained variance from preprocessed trajectory vectors.

Builds per-flight vectors in the canonical interleaved format:
`[x1, y1, x2, y2, ..., xM, yM]` using `x_utm` and `y_utm`.

For variable-length trajectories, pass `--resample-points` to enforce
a common number of points per flight before PCA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA


def _resample_xy(coords: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a 2D trajectory to ``n_points`` on normalized arc index."""

    if coords.shape[0] == n_points:
        return coords
    if coords.shape[0] == 1:
        return np.repeat(coords, n_points, axis=0)

    src = np.linspace(0.0, 1.0, num=coords.shape[0])
    dst = np.linspace(0.0, 1.0, num=n_points)
    x = np.interp(dst, src, coords[:, 0])
    y = np.interp(dst, src, coords[:, 1])
    return np.column_stack([x, y])


def _interleaved_vector(coords: np.ndarray) -> np.ndarray:
    """Return `[x1,y1,x2,y2,...]` vector for one trajectory."""

    vec = np.empty(coords.shape[0] * 2, dtype=float)
    vec[0::2] = coords[:, 0]
    vec[1::2] = coords[:, 1]
    return vec


def _iter_flights(df: pd.DataFrame) -> Iterable[tuple[int, np.ndarray, str | None, str | None]]:
    for fid, grp in df.groupby("flight_id", sort=True):
        coords = grp.sort_values("step")[["x_utm", "y_utm"]].to_numpy(dtype=float)
        if coords.shape[0] >= 2:
            ad = str(grp["A/D"].iloc[0]) if "A/D" in grp.columns else None
            runway = str(grp["Runway"].iloc[0]) if "Runway" in grp.columns else None
            yield int(fid), coords, ad, runway


def _sample_plot_df(df: pd.DataFrame, max_points: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=max_points, replace=False)
    return df.loc[np.sort(idx)].copy()


def _plot_pc_scatter(df_scores: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    has_flow = "flow_label" in df_scores.columns and df_scores["flow_label"].nunique(dropna=True) > 1

    if has_flow:
        uniq = sorted(df_scores["flow_label"].dropna().unique().tolist())
        cmap = plt.get_cmap("tab10")
        color_map = {label: cmap(i % 10) for i, label in enumerate(uniq)}
        for label in uniq:
            part = df_scores[df_scores["flow_label"] == label]
            ax.scatter(part[x_col], part[y_col], s=9, alpha=0.45, color=color_map[label], label=label)
        ax.legend(loc="best", fontsize=8, frameon=True)
    else:
        ax.scatter(df_scores[x_col], df_scores[y_col], s=9, alpha=0.45, color="#4E79A7")

    ax.axhline(0.0, color="#D0D0D0", linewidth=0.8)
    ax.axvline(0.0, color="#D0D0D0", linewidth=0.8)
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())
    ax.set_title(title)
    ax.grid(True, color="#EAEAEA", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_pairplot(df_scores: pd.DataFrame, pc_cols: list[str], out_path: Path) -> None:
    if len(pc_cols) < 2:
        return
    axes = scatter_matrix(
        df_scores[pc_cols],
        alpha=0.35,
        diagonal="hist",
        figsize=(10, 10),
        color="#4E79A7",
    )
    fig = axes[0, 0].figure
    fig.suptitle("PCA score pairplot", y=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_pca(
    input_csv: Path,
    output_dir: Path,
    report_components: int = 5,
    n_components: int = 20,
    resample_points: int | None = None,
    max_flights: int | None = None,
    flow: str | None = None,
    max_plot_points: int = 5000,
) -> dict:
    usecols = ["flight_id", "step", "x_utm", "y_utm", "A/D", "Runway"]
    df = pd.read_csv(input_csv, usecols=[c for c in usecols if c in pd.read_csv(input_csv, nrows=0).columns])

    if flow:
        ad, runway = flow.split("_", 1)
        if "A/D" not in df.columns or "Runway" not in df.columns:
            raise ValueError("Flow filter requested, but A/D and Runway columns are missing.")
        df = df[(df["A/D"] == ad) & (df["Runway"] == runway)]

    vectors: list[np.ndarray] = []
    flight_ids: list[int] = []
    flight_ads: list[str | None] = []
    flight_runways: list[str | None] = []
    flight_flow_labels: list[str | None] = []
    lengths: list[int] = []
    for fid, coords, ad, runway in _iter_flights(df):
        lengths.append(coords.shape[0])
        if resample_points is not None:
            coords = _resample_xy(coords, resample_points)
        vectors.append(_interleaved_vector(coords))
        flight_ids.append(fid)
        flight_ads.append(ad)
        flight_runways.append(runway)
        if ad is not None and runway is not None:
            flight_flow_labels.append(f"{ad}_{runway}")
        else:
            flight_flow_labels.append(None)
        if max_flights is not None and len(vectors) >= max_flights:
            break

    if not vectors:
        raise ValueError("No valid flights found for PCA.")

    unique_lens = sorted(set(lengths))
    if resample_points is None and len(unique_lens) != 1:
        raise ValueError(
            "Variable-length trajectories found. Re-run with --resample-points "
            f"(unique lengths={unique_lens[:10]}{'...' if len(unique_lens) > 10 else ''})."
        )

    X = np.vstack(vectors)
    n_components_eff = int(min(n_components, X.shape[0], X.shape[1]))
    pca = PCA(n_components=n_components_eff, svd_solver="full")
    scores = pca.fit_transform(X)

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    report_n = int(min(report_components, len(ratios)))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_csv.stem + (f"_{flow}" if flow else "")
    out_csv = output_dir / f"{stem}_pca_variance.csv"
    out_json = output_dir / f"{stem}_pca_summary.json"
    out_png = output_dir / f"{stem}_pca_scree.png"
    out_scores_csv = output_dir / f"{stem}_pca_scores.csv"
    out_pc12_png = output_dir / f"{stem}_pca_scores_pc1_pc2.png"
    out_pc13_png = output_dir / f"{stem}_pca_scores_pc1_pc3.png"
    out_pair_png = output_dir / f"{stem}_pca_scores_pairplot_top4.png"

    df_var = pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1, dtype=int),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )
    df_var.to_csv(out_csv, index=False)

    score_cols = [f"pc{i + 1}" for i in range(scores.shape[1])]
    df_scores = pd.DataFrame(scores, columns=score_cols)
    df_scores.insert(0, "flight_id", flight_ids)
    if any(v is not None for v in flight_ads):
        df_scores.insert(1, "A/D", flight_ads)
    if any(v is not None for v in flight_runways):
        insert_at = 2 if "A/D" in df_scores.columns else 1
        df_scores.insert(insert_at, "Runway", flight_runways)
    if any(v is not None for v in flight_flow_labels):
        insert_at = 3 if "Runway" in df_scores.columns and "A/D" in df_scores.columns else 1
        df_scores.insert(insert_at, "flow_label", flight_flow_labels)
    df_scores.to_csv(out_scores_csv, index=False)

    df_plot = _sample_plot_df(df_scores, max_points=max_plot_points, seed=42)
    if "pc1" in df_plot.columns and "pc2" in df_plot.columns:
        _plot_pc_scatter(
            df_plot,
            "pc1",
            "pc2",
            title=f"PCA scores PC1 vs PC2 ({input_csv.name})" + (f" [{flow}]" if flow else ""),
            out_path=out_pc12_png,
        )
    if "pc1" in df_plot.columns and "pc3" in df_plot.columns:
        _plot_pc_scatter(
            df_plot,
            "pc1",
            "pc3",
            title=f"PCA scores PC1 vs PC3 ({input_csv.name})" + (f" [{flow}]" if flow else ""),
            out_path=out_pc13_png,
        )
    pair_cols = [c for c in ("pc1", "pc2", "pc3", "pc4") if c in df_plot.columns]
    _plot_pairplot(df_plot, pair_cols, out_pair_png)

    summary = {
        "input_csv": str(input_csv),
        "flow": flow,
        "n_flights_used": int(X.shape[0]),
        "vector_dim": int(X.shape[1]),
        "resample_points": int(resample_points) if resample_points is not None else None,
        "lengths_min": int(min(lengths)),
        "lengths_median": int(np.median(lengths)),
        "lengths_max": int(max(lengths)),
        "report_components": report_n,
        "first_components": [
            {
                "component": int(i + 1),
                "explained_variance_ratio": float(ratios[i]),
                "cumulative_explained_variance_ratio": float(cumulative[i]),
            }
            for i in range(report_n)
        ],
        "csv_output": str(out_csv),
        "plot_output": str(out_png),
        "scores_csv_output": str(out_scores_csv),
        "scores_plot_pc12_output": str(out_pc12_png),
        "scores_plot_pc13_output": str(out_pc13_png),
        "scores_pairplot_output": str(out_pair_png),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Scree + cumulative plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    comps = np.arange(1, len(ratios) + 1)
    ax1.bar(comps, ratios, alpha=0.6, label="Explained variance ratio")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_title(f"PCA on {input_csv.name}" + (f" ({flow})" if flow else ""))

    ax2 = ax1.twinx()
    ax2.plot(comps, cumulative, color="black", linewidth=2, marker="o", markersize=3, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA explained variance for preprocessed trajectories.")
    parser.add_argument("--input-csv", required=True, type=Path, help="Preprocessed CSV path.")
    parser.add_argument("--output-dir", default=Path("output/eda/pca_preprocessed"), type=Path)
    parser.add_argument("--report-components", default=5, type=int)
    parser.add_argument("--n-components", default=20, type=int)
    parser.add_argument("--resample-points", default=None, type=int)
    parser.add_argument("--max-flights", default=None, type=int)
    parser.add_argument("--flow", default=None, type=str, help="Flow filter, e.g. Landung_09L")
    parser.add_argument("--max-plot-points", default=5000, type=int)
    args = parser.parse_args()

    summary = run_pca(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        report_components=args.report_components,
        n_components=args.n_components,
        resample_points=args.resample_points,
        max_flights=args.max_flights,
        flow=args.flow,
        max_plot_points=args.max_plot_points,
    )

    print("PCA summary:")
    for row in summary["first_components"]:
        print(
            f"PC{row['component']}: "
            f"explained={row['explained_variance_ratio']:.6f} "
            f"cumulative={row['cumulative_explained_variance_ratio']:.6f}"
        )
    print(f"Saved CSV: {summary['csv_output']}")
    print(f"Saved plot: {summary['plot_output']}")
    print(f"Saved scores: {summary['scores_csv_output']}")
    print(f"Saved score plot PC1-PC2: {summary['scores_plot_pc12_output']}")
    if Path(summary["scores_plot_pc13_output"]).exists():
        print(f"Saved score plot PC1-PC3: {summary['scores_plot_pc13_output']}")
    if Path(summary["scores_pairplot_output"]).exists():
        print(f"Saved score pairplot: {summary['scores_pairplot_output']}")


if __name__ == "__main__":
    main()
