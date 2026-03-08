"""Velocity leakage diagnostic for trajectory clustering distances.

Given flight trajectories as 2D point sequences, this script compares:

1) Index-aligned Euclidean distances:
   d_index(A,B) = sqrt(sum_i ||p_A[i] - p_B[i]||_2^2)
   where trajectories are first resampled to a fixed length L.

2) Alignment-aware DTW distances:
   d_dtw(A,B) = min_{pi in Pi} sum_{(i,j) in pi} ||p_A[i] - p_B[j]||_2

3) Correlation between upper-triangle vectors:
   u = vec_triangle(D_index), v = vec_triangle(D_dtw)
   rho = corr(u, v) (Pearson + Spearman)

Low correlation (default threshold: rho_pearson < 0.6) suggests index-based
distances are sensitive to speed/time-parameterization mismatch.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class FlightTrajectory:
    flight_id: Any
    idx: np.ndarray  # shape (n,)
    xy: np.ndarray  # shape (n, 2)


def _print(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def _detect_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext == ".csv":
        return "csv"
    if ext == ".parquet":
        return "parquet"
    if ext in {".pkl", ".pickle", ".joblib"}:
        return "pickle"
    raise ValueError(f"Cannot infer format from extension '{ext}'. Pass --format explicitly.")


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_points(idx: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    idx = np.asarray(idx, dtype=float).reshape(-1)
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return None
    xy = xy[:, :2]

    mask = np.isfinite(idx) & np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    idx = idx[mask]
    xy = xy[mask]
    if idx.size < 2:
        return None

    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    xy = xy[order]

    # Merge duplicate index values by averaging coordinates.
    uniq, inv = np.unique(idx, return_inverse=True)
    if uniq.size != idx.size:
        sums = np.zeros((uniq.size, 2), dtype=float)
        counts = np.zeros(uniq.size, dtype=float)
        np.add.at(sums, inv, xy)
        np.add.at(counts, inv, 1.0)
        xy = sums / counts[:, None]
        idx = uniq

    if idx.size < 2:
        return None
    return idx, xy


def _table_to_flights(df: pd.DataFrame) -> tuple[list[FlightTrajectory], int]:
    cols = list(df.columns)
    flight_col = _find_col(cols, ["flight_id"])
    idx_col = _find_col(cols, ["idx", "step"])
    x_col = _find_col(cols, ["x", "x_utm"])
    y_col = _find_col(cols, ["y", "y_utm"])

    if not all([flight_col, idx_col, x_col, y_col]):
        raise ValueError(
            "Tabular input requires columns: flight_id, idx|step, x|x_utm, y|y_utm."
        )

    work = df[[flight_col, idx_col, x_col, y_col]].copy()
    work[idx_col] = pd.to_numeric(work[idx_col], errors="coerce")
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work[work[flight_col].notna()]

    flights: list[FlightTrajectory] = []
    dropped = 0
    for fid, grp in work.groupby(flight_col, sort=True):
        norm = _normalize_points(
            grp[idx_col].to_numpy(dtype=float),
            grp[[x_col, y_col]].to_numpy(dtype=float),
        )
        if norm is None:
            dropped += 1
            continue
        idx, xy = norm
        flights.append(FlightTrajectory(flight_id=fid, idx=idx, xy=xy))
    return flights, dropped


def _mapping_to_flights(mapping: dict[Any, Any]) -> tuple[list[FlightTrajectory], int]:
    flights: list[FlightTrajectory] = []
    dropped = 0
    for fid, val in mapping.items():
        arr = np.asarray(val)
        if arr.ndim != 2 or arr.shape[1] < 2:
            dropped += 1
            continue
        idx = np.arange(arr.shape[0], dtype=float)
        norm = _normalize_points(idx, arr[:, :2].astype(float))
        if norm is None:
            dropped += 1
            continue
        idx_n, xy_n = norm
        flights.append(FlightTrajectory(flight_id=fid, idx=idx_n, xy=xy_n))
    return flights, dropped


def _load_pickle_or_joblib(path: Path, verbose: bool) -> tuple[list[FlightTrajectory], int]:
    obj: Any | None = None
    last_err: Exception | None = None

    if path.suffix.lower() == ".joblib":
        try:
            import joblib

            obj = joblib.load(path)
        except Exception as exc:  # pragma: no cover - env dependent
            last_err = exc
    else:
        try:
            with path.open("rb") as fh:
                obj = pickle.load(fh)
        except Exception as exc:
            last_err = exc
            try:
                import joblib

                obj = joblib.load(path)
                last_err = None
            except Exception as exc2:  # pragma: no cover - env dependent
                last_err = exc2

    if obj is None:
        raise ValueError(f"Failed to load pickle/joblib input: {last_err}")

    if isinstance(obj, dict):
        _print("Loaded dict-like payload from pickle/joblib.", verbose)
        return _mapping_to_flights(obj)
    if isinstance(obj, pd.DataFrame):
        _print("Loaded DataFrame payload from pickle/joblib.", verbose)
        return _table_to_flights(obj)
    raise ValueError("Unsupported pickle/joblib payload. Expected dict or pandas DataFrame.")


def _parquet_payload_to_mapping(df: pd.DataFrame) -> dict[Any, Any] | None:
    if df.shape == (1, 1):
        obj = df.iat[0, 0]
        if isinstance(obj, dict):
            return obj

    # Best effort: first non-null object cell containing a dict.
    for col in df.columns:
        if df[col].dtype == "object":
            series = df[col].dropna()
            if not series.empty and isinstance(series.iloc[0], dict):
                return series.iloc[0]
    return None


def _load_parquet(path: Path, verbose: bool) -> tuple[list[FlightTrajectory], int]:
    df = pd.read_parquet(path)

    # Try table semantics first.
    try:
        flights, dropped = _table_to_flights(df)
        _print("Parsed parquet as tabular trajectories.", verbose)
        return flights, dropped
    except Exception:
        pass

    # Try object payload semantics.
    mapping = _parquet_payload_to_mapping(df)
    if mapping is not None:
        _print("Parsed parquet as dict payload.", verbose)
        return _mapping_to_flights(mapping)

    raise ValueError("Unsupported parquet structure for trajectory loading.")


def _load_csv(path: Path, verbose: bool) -> tuple[list[FlightTrajectory], int]:
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    flight_col = _find_col(cols, ["flight_id"])
    idx_col = _find_col(cols, ["idx", "step"])
    x_col = _find_col(cols, ["x", "x_utm"])
    y_col = _find_col(cols, ["y", "y_utm"])

    if not all([flight_col, idx_col, x_col, y_col]):
        raise ValueError(
            "CSV must contain flight_id and idx|step and x|x_utm and y|y_utm columns."
        )

    usecols = [flight_col, idx_col, x_col, y_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    _print(
        f"CSV columns used: flight='{flight_col}', idx='{idx_col}', x='{x_col}', y='{y_col}'",
        verbose,
    )
    return _table_to_flights(df)


def load_flights(path: Path, fmt: str, verbose: bool) -> tuple[list[FlightTrajectory], int, str]:
    resolved_fmt = _detect_format(path, fmt)
    if resolved_fmt == "csv":
        flights, dropped = _load_csv(path, verbose)
    elif resolved_fmt == "pickle":
        flights, dropped = _load_pickle_or_joblib(path, verbose)
    elif resolved_fmt == "parquet":
        flights, dropped = _load_parquet(path, verbose)
    else:
        raise ValueError(f"Unsupported format: {resolved_fmt}")
    return flights, dropped, resolved_fmt


def _resample_by_index(idx: np.ndarray, xy: np.ndarray, L: int) -> np.ndarray:
    if L < 2:
        raise ValueError("L must be >= 2.")
    start = float(idx[0])
    end = float(idx[-1])
    if end == start:
        return np.repeat(xy[[0]], L, axis=0)
    tgt = np.linspace(start, end, L, dtype=float)
    x = np.interp(tgt, idx, xy[:, 0])
    y = np.interp(tgt, idx, xy[:, 1])
    return np.column_stack([x, y])


def _make_progress(label: str, total: int, verbose: bool):
    if not verbose or total <= 0:
        return lambda delta: None

    start = time.perf_counter()
    step = max(1000, total // 100 if total >= 100 else 1)
    state = {"done": 0, "last": 0}

    def update(delta: int) -> None:
        state["done"] += int(delta)
        done = state["done"]
        if done - state["last"] >= step or done >= total:
            elapsed = max(time.perf_counter() - start, 1e-9)
            rate = done / elapsed
            pct = 100.0 * done / total
            print(
                f"[{label}] {done}/{total} pairs ({pct:.1f}%) "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} pairs/s",
                flush=True,
            )
            state["last"] = done

    return update


def build_index_distance_matrix(flights: list[FlightTrajectory], L: int, verbose: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    resampled = [_resample_by_index(f.idx, f.xy, L) for f in flights]
    X = np.vstack([arr.reshape(1, -1) for arr in resampled])

    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    total_pairs = n * (n - 1) // 2
    progress = _make_progress("D_index", total_pairs, verbose)

    for i in range(n - 1):
        diff = X[i + 1 :] - X[i]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        D[i, i + 1 :] = d
        D[i + 1 :, i] = d
        progress(len(d))

    np.fill_diagonal(D, 0.0)
    return D, resampled


def dtw_distance_2d(a: np.ndarray, b: np.ndarray, window: int | None = None) -> float:
    n = int(a.shape[0])
    m = int(b.shape[0])
    if n == 0 or m == 0:
        return float("inf")

    if window is None:
        w = max(n, m)
    else:
        if window < 0:
            raise ValueError("dtw_window must be non-negative.")
        w = max(int(window), abs(n - m))

    prev = np.full(m + 1, np.inf, dtype=float)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf, dtype=float)
        ai = a[i - 1]
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return float(prev[m])


def build_dtw_distance_matrix(
    trajectories: list[np.ndarray],
    dtw_window: int | None,
    verbose: bool,
) -> np.ndarray:
    n = len(trajectories)
    D = np.zeros((n, n), dtype=float)
    total_pairs = n * (n - 1) // 2
    progress = _make_progress("D_dtw", total_pairs, verbose)

    for i in range(n - 1):
        ai = trajectories[i]
        for j in range(i + 1, n):
            d = dtw_distance_2d(ai, trajectories[j], window=dtw_window)
            D[i, j] = d
            D[j, i] = d
            progress(1)

    np.fill_diagonal(D, 0.0)
    return D


def _upper_triangle_values(D: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(D, k=1)
    return D[idx]


def _safe_corr(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, float]:
    pearson_r = float("nan")
    pearson_p = float("nan")
    spearman_r = float("nan")
    spearman_p = float("nan")

    try:
        pr = pearsonr(u, v)
        pearson_r = float(getattr(pr, "statistic", pr[0]))
        pearson_p = float(getattr(pr, "pvalue", pr[1]))
    except Exception:
        pass

    try:
        sr = spearmanr(u, v)
        spearman_r = float(getattr(sr, "statistic", sr[0]))
        spearman_p = float(getattr(sr, "pvalue", sr[1]))
    except Exception:
        pass

    return pearson_r, pearson_p, spearman_r, spearman_p


def _interpret(rho_pearson: float, threshold: float = 0.6) -> str:
    if not np.isfinite(rho_pearson):
        return "Pearson correlation is undefined (likely near-constant distance vectors)."
    if rho_pearson < threshold:
        return "Index-based distances likely sensitive to speed/misalignment."
    return "Index-based and alignment-aware geometry are broadly consistent."


def _jsonable_id(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=8, alpha=0.25, color="#4E79A7")
    ax.set_xlabel("D_index (upper triangle)")
    ax.set_ylabel("D_dtw (upper triangle)")
    ax.set_title(title)
    ax.grid(True, color="#EAEAEA", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _run_self_test(verbose: bool) -> int:
    _print("Running self-test...", True)
    n = 80
    t = np.linspace(0.0, 1.0, n)
    base = np.column_stack([1000.0 * t, 120.0 * np.sin(2.0 * np.pi * t)])

    def from_param(u: np.ndarray) -> np.ndarray:
        x = np.interp(u, t, base[:, 0])
        y = np.interp(u, t, base[:, 1])
        return np.column_stack([x, y])

    # Warped set: same shape, different parameterizations + some shifted variants.
    warped = []
    for power in [1.0, 1.6, 0.6, 2.0, 0.4]:
        u = t**power
        warped.append(from_param(u))
    for shift in [0.0, 20.0, -25.0, 40.0, -35.0]:
        warped.append(base + np.array([0.0, shift]))

    # Control set: same parameterization, mostly rigid offsets.
    control = [base + np.array([0.0, s]) for s in [0.0, 20.0, -25.0, 40.0, -35.0, 10.0, -10.0, 30.0, -30.0, 15.0]]

    def to_flights(trajs: list[np.ndarray]) -> list[FlightTrajectory]:
        out = []
        for i, arr in enumerate(trajs):
            out.append(FlightTrajectory(flight_id=f"f{i}", idx=np.arange(arr.shape[0], dtype=float), xy=arr))
        return out

    fw = to_flights(warped)
    Dw_idx, rs_w = build_index_distance_matrix(fw, L=40, verbose=verbose)
    Dw_dtw = build_dtw_distance_matrix([f.xy for f in fw], dtw_window=None, verbose=verbose)

    assert np.allclose(Dw_idx, Dw_idx.T), "Index matrix not symmetric."
    assert np.allclose(Dw_dtw, Dw_dtw.T), "DTW matrix not symmetric."
    assert np.allclose(np.diag(Dw_idx), 0.0), "Index matrix diagonal not zero."
    assert np.allclose(np.diag(Dw_dtw), 0.0), "DTW matrix diagonal not zero."
    assert np.isfinite(Dw_idx).all() and np.isfinite(Dw_dtw).all(), "Non-finite values in matrices."

    # Base vs warped should be cheaper under DTW than index-aligned distance.
    assert Dw_dtw[0, 1] < Dw_idx[0, 1], "Expected DTW to better align time-warped trajectories."

    u_w = _upper_triangle_values(Dw_idx)
    v_w = _upper_triangle_values(Dw_dtw)
    rho_w, _, _, _ = _safe_corr(u_w, v_w)

    fc = to_flights(control)
    Dc_idx, _ = build_index_distance_matrix(fc, L=40, verbose=verbose)
    Dc_dtw = build_dtw_distance_matrix([f.xy for f in fc], dtw_window=None, verbose=verbose)
    u_c = _upper_triangle_values(Dc_idx)
    v_c = _upper_triangle_values(Dc_dtw)
    rho_c, _, _, _ = _safe_corr(u_c, v_c)

    assert np.isfinite(rho_w), "Warped Pearson correlation is undefined."
    assert np.isfinite(rho_c), "Control Pearson correlation is undefined."
    assert rho_w < 0.6, f"Warped set expected low correlation, got {rho_w:.4f}"
    assert rho_c > 0.9, f"Control set expected high correlation, got {rho_c:.4f}"

    _print(f"Self-test passed. warped_rho={rho_w:.4f}, control_rho={rho_c:.4f}", True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose velocity/time-parameterization leakage via index-distance vs DTW-distance correlation."
    )
    parser.add_argument("--input", type=Path, required=False, help="Input trajectory dataset path.")
    parser.add_argument("--format", choices=["auto", "csv", "pickle", "parquet"], default="auto")
    parser.add_argument("--L", type=int, default=40, help="Resample length for index-based distances.")
    parser.add_argument("--max_flights", type=int, default=None, help="Max flights to use (random subsample).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling.")
    parser.add_argument("--dtw_window", type=int, default=None, help="Sakoe-Chiba DTW window (index units).")
    parser.add_argument("--dtw_source", choices=["original", "resampled"], default="original")
    parser.add_argument("--save_scatter", action="store_true", help="Save D_index vs D_dtw scatter plot.")
    parser.add_argument("--self_test", action="store_true", help="Run built-in synthetic checks and exit.")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress/logging.")
    args = parser.parse_args()

    if args.self_test:
        code = _run_self_test(verbose=args.verbose)
        raise SystemExit(code)

    if args.input is None:
        raise ValueError("--input is required unless --self_test is used.")
    if args.L < 2:
        raise ValueError("--L must be >= 2.")
    if args.max_flights is not None and args.max_flights < 3:
        raise ValueError("--max_flights must be >= 3 when provided.")

    path = args.input
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    flights, dropped_cleaning, resolved_fmt = load_flights(path, args.format, args.verbose)
    if not flights:
        raise RuntimeError("No valid flights after loading/cleaning.")

    if args.max_flights is not None and len(flights) > args.max_flights:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(flights), size=args.max_flights, replace=False)
        flights = [flights[i] for i in sorted(sel)]
        _print(f"Subsampled to {len(flights)} flights (seed={args.seed}).", True)

    n_flights = len(flights)
    if n_flights < 3:
        raise RuntimeError(f"Need at least 3 flights for correlation; got {n_flights}.")

    _print(f"Flights used (N): {n_flights}", True)
    _print(f"Resample length (L): {args.L}", True)
    _print(f"DTW source: {args.dtw_source}", True)
    if args.dtw_window is not None:
        _print(f"DTW window: {args.dtw_window}", True)

    D_index, resampled = build_index_distance_matrix(flights, L=args.L, verbose=args.verbose)
    if args.dtw_source == "resampled":
        dtw_trajs = resampled
    else:
        dtw_trajs = [f.xy for f in flights]
    D_dtw = build_dtw_distance_matrix(dtw_trajs, dtw_window=args.dtw_window, verbose=args.verbose)

    # Enforce symmetry and zero diagonals defensively.
    D_index = 0.5 * (D_index + D_index.T)
    D_dtw = 0.5 * (D_dtw + D_dtw.T)
    np.fill_diagonal(D_index, 0.0)
    np.fill_diagonal(D_dtw, 0.0)

    u = _upper_triangle_values(D_index)
    v = _upper_triangle_values(D_dtw)
    rho_p, p_p, rho_s, p_s = _safe_corr(u, v)
    interpretation = _interpret(rho_p, threshold=0.6)

    _print(f"Pearson rho={rho_p:.6f}, p={p_p:.6g}", True)
    _print(f"Spearman rho={rho_s:.6f}, p={p_s:.6g}", True)
    _print(
        f"D_index mean={float(np.mean(u)):.6f}, median={float(np.median(u)):.6f}",
        True,
    )
    _print(
        f"D_dtw   mean={float(np.mean(v)):.6f}, median={float(np.median(v)):.6f}",
        True,
    )
    _print(f"Interpretation: {interpretation}", True)

    out_json = path.with_name(f"{path.stem}_velocity_leakage_diagnostic.json")
    result = {
        "input_path": str(path),
        "format": resolved_fmt,
        "N": int(n_flights),
        "L": int(args.L),
        "dtw_source": args.dtw_source,
        "dtw_window": int(args.dtw_window) if args.dtw_window is not None else None,
        "pairs": int(u.size),
        "rho_pearson": float(rho_p),
        "p_pearson": float(p_p),
        "rho_spearman": float(rho_s),
        "p_spearman": float(p_s),
        "d_index_mean": float(np.mean(u)),
        "d_index_median": float(np.median(u)),
        "d_dtw_mean": float(np.mean(v)),
        "d_dtw_median": float(np.median(v)),
        "interpretation": interpretation,
        "dropped_flights_too_short_or_nan": int(dropped_cleaning),
        "used_flight_ids": [_jsonable_id(f.flight_id) for f in flights],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print(f"Saved JSON: {out_json}", True)

    if args.save_scatter:
        out_png = path.with_name(f"{path.stem}_velocity_leakage_scatter.png")
        _save_scatter(
            out_png,
            u,
            v,
            title=f"Velocity leakage diagnostic ({path.name})",
        )
        _print(f"Saved scatter: {out_png}", True)


if __name__ == "__main__":
    main()
