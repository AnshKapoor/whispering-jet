"""Velocity leakage audit for ADS-B trajectory clustering.

This script quantifies whether trajectory time-parameterization (velocity effects)
changes clustering outcomes by comparing two trajectory representations:

1) Time-index (or uniform-time for raw trajectories with timestamps)
2) Arc-length reparameterized

For each dataset/target-length L, it computes:
- ARI(labels_time, labels_arclen)
- Distance inflation ratio r_ij = d_time(i,j) / (d_arclen(i,j) + eps)
- Optional silhouette for each representation

CLI:
    python velocity_leakage_audit.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import scoreatpercentile
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class RunContext:
    out_dir: Path
    logs: list[str]


def _log(ctx: RunContext, message: str) -> None:
    line = str(message)
    print(line, flush=True)
    ctx.logs.append(line)


def _to_float_time(values: pd.Series) -> np.ndarray:
    """Convert a time-like series to float seconds when possible."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() >= max(2, int(0.8 * len(values))):
        return numeric.to_numpy(dtype=float)

    dt = pd.to_datetime(values, errors="coerce", utc=True)
    out = np.full(len(values), np.nan, dtype=float)
    mask = dt.notna().to_numpy()
    if mask.any():
        out[mask] = (dt[mask].astype("int64") / 1e9).to_numpy(dtype=float)
    return out


def _detect_table_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = list(df.columns)

    def pick(candidates: list[str]) -> str | None:
        lc_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lc_map:
                return lc_map[cand.lower()]
        return None

    return {
        "traj_id": pick(["traj_id", "trajectory_id", "flight_id", "id"]),
        "t": pick(["t", "time", "timestamp", "idx", "step"]),
        "x": pick(["x", "x_utm", "longitude", "lon"]),
        "y": pick(["y", "y_utm", "latitude", "lat"]),
        "z": pick(["z", "alt", "altitude", "geoalt"]),
    }


def clean_trajectory(
    traj: dict[str, Any],
    dims: int,
    ctx: RunContext | None = None,
) -> dict[str, Any] | None:
    """Clean one trajectory: finite rows only, remove consecutive duplicate points."""
    p = np.asarray(traj["p"], dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        return None
    if p.shape[1] < dims:
        return None
    p = p[:, :dims]

    t_raw = traj.get("t")
    t: np.ndarray | None = None
    if t_raw is not None:
        t = np.asarray(t_raw, dtype=float).reshape(-1)
        if t.shape[0] != p.shape[0]:
            t = None

    mask = np.isfinite(p).all(axis=1)
    if t is not None:
        mask &= np.isfinite(t)
    p = p[mask]
    if t is not None:
        t = t[mask]

    if p.shape[0] < 2:
        return None

    # Remove consecutive duplicates in geometry.
    deltas = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.hstack([[True], deltas > 0.0])
    p = p[keep]
    if t is not None:
        t = t[keep]

    if p.shape[0] < 2:
        return None

    return {"id": traj["id"], "t": t, "p": p}


def _interp_by_param(param: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linear interpolation for multivariate values over a 1D parameter."""
    param = np.asarray(param, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)

    order = np.argsort(param, kind="mergesort")
    param = param[order]
    values = values[order]

    uniq, uniq_idx = np.unique(param, return_index=True)
    param = uniq
    values = values[uniq_idx]

    if param.shape[0] == 1:
        return np.repeat(values, target.shape[0], axis=0)

    out = np.zeros((target.shape[0], values.shape[1]), dtype=float)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(target, param, values[:, d])
    return out


def resample_uniform_time(traj: dict[str, Any], target_L: int) -> np.ndarray:
    """Resample trajectory uniformly in time; fallback to index when no valid time exists."""
    p = traj["p"]
    t = traj.get("t")
    n = p.shape[0]

    if target_L <= 1:
        return p[[0]].copy()

    if t is not None and np.isfinite(t).all() and np.nanmax(t) > np.nanmin(t):
        param = t.astype(float)
    else:
        param = np.linspace(0.0, 1.0, n)
    target = np.linspace(float(np.nanmin(param)), float(np.nanmax(param)), target_L)
    return _interp_by_param(param, p, target)


def _resample_uniform_index(traj: dict[str, Any], target_L: int) -> np.ndarray:
    p = traj["p"]
    n = p.shape[0]
    if target_L <= 1:
        return p[[0]].copy()
    param = np.linspace(0.0, 1.0, n)
    target = np.linspace(0.0, 1.0, target_L)
    return _interp_by_param(param, p, target)


def resample_arclength(traj: dict[str, Any], target_L: int) -> tuple[np.ndarray, bool]:
    """Resample uniformly in arc-length. Returns (points, is_constant_geometry)."""
    p = traj["p"]
    if target_L <= 1:
        return p[[0]].copy(), False

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.hstack([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    if total <= 0.0:
        return np.repeat(p[[0]], target_L, axis=0), True

    target = np.linspace(0.0, total, target_L)
    return _interp_by_param(s, p, target), False


def vectorize(points_list: list[np.ndarray]) -> np.ndarray:
    """Flatten trajectory points into feature vectors (N, L*D)."""
    if not points_list:
        return np.zeros((0, 0), dtype=float)
    return np.vstack([p.reshape(1, -1) for p in points_list])


def cluster(
    X: np.ndarray,
    cfg: dict[str, Any],
    random_state: int,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Cluster standardized features according to config."""
    method = str(cfg.get("method", "kmeans")).strip().lower()
    k = cfg.get("k")
    standardize = bool(cfg.get("standardize", True))

    if X.shape[0] == 0:
        return np.array([], dtype=int), method, X

    X_fit = X
    if standardize and X.shape[1] > 0:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    if method == "kmeans":
        if k is None:
            raise ValueError("clustering.k must be provided for kmeans")
        model = KMeans(n_clusters=int(k), random_state=int(random_state), n_init=20)
        labels = model.fit_predict(X_fit)
        return labels.astype(int), "kmeans", X_fit

    if method == "agglomerative":
        if k is None:
            raise ValueError("clustering.k must be provided for agglomerative")
        model = AgglomerativeClustering(n_clusters=int(k))
        labels = model.fit_predict(X_fit)
        return labels.astype(int), "agglomerative", X_fit

    if method == "hdbscan_if_available":
        try:
            import hdbscan  # type: ignore

            hcfg = cfg.get("hdbscan", {}) or {}
            model = hdbscan.HDBSCAN(
                min_cluster_size=int(hcfg.get("min_cluster_size", 10)),
                min_samples=None if hcfg.get("min_samples") is None else int(hcfg.get("min_samples")),
                allow_single_cluster=bool(hcfg.get("allow_single_cluster", False)),
            )
            labels = model.fit_predict(X_fit)
            return labels.astype(int), "hdbscan", X_fit
        except Exception:
            # deterministic fallback to kmeans
            k_fb = int(k) if k is not None else 2
            model = KMeans(n_clusters=k_fb, random_state=int(random_state), n_init=20)
            labels = model.fit_predict(X_fit)
            return labels.astype(int), "kmeans_fallback", X_fit

    raise ValueError(f"Unsupported clustering method: {method}")


def _sample_pairs(n: int, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to m unique unordered pairs (i,j), i<j."""
    total = n * (n - 1) // 2
    if n < 2 or total == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if m >= total:
        i, j = np.triu_indices(n, k=1)
        return i.astype(int), j.astype(int)

    pairs: set[tuple[int, int]] = set()
    target = int(m)
    while len(pairs) < target:
        batch = min((target - len(pairs)) * 3, 200000)
        a = rng.integers(0, n, size=batch, endpoint=False)
        b = rng.integers(0, n, size=batch, endpoint=False)
        mask = a < b
        if not np.any(mask):
            continue
        for x, y in zip(a[mask], b[mask]):
            pairs.add((int(x), int(y)))
            if len(pairs) >= target:
                break

    arr = np.array(list(pairs), dtype=int)
    return arr[:, 0], arr[:, 1]


def compute_metrics(
    X_time: np.ndarray,
    X_arclen: np.ndarray,
    labels_time: np.ndarray,
    labels_arclen: np.ndarray,
    pairs_sample: int,
    eps: float,
    rng: np.random.Generator,
    allow_silhouette: bool,
) -> dict[str, Any]:
    """Compute ARI + distance inflation + optional silhouette metrics."""
    n = int(X_time.shape[0])
    metrics: dict[str, Any] = {
        "N": n,
        "ARI": float(adjusted_rand_score(labels_time, labels_arclen)) if n > 1 else np.nan,
    }

    i, j = _sample_pairs(n, int(pairs_sample), rng)
    if i.size:
        dt = np.linalg.norm(X_time[i] - X_time[j], axis=1)
        ds = np.linalg.norm(X_arclen[i] - X_arclen[j], axis=1)
        r = dt / (ds + float(eps))
        metrics.update(
            {
                "pairs_used": int(i.size),
                "r_median": float(np.median(r)),
                "r_p90": float(scoreatpercentile(r, 90)),
                "r_p95": float(scoreatpercentile(r, 95)),
                "r_mean": float(np.mean(r)),
            }
        )
    else:
        metrics.update(
            {
                "pairs_used": 0,
                "r_median": np.nan,
                "r_p90": np.nan,
                "r_p95": np.nan,
                "r_mean": np.nan,
            }
        )

    def sil(X: np.ndarray, labels: np.ndarray) -> float:
        uniq = np.unique(labels)
        if uniq.size <= 1:
            return np.nan
        return float(silhouette_score(X, labels))

    if allow_silhouette:
        try:
            metrics["silhouette_time"] = sil(X_time, labels_time)
        except Exception:
            metrics["silhouette_time"] = np.nan
        try:
            metrics["silhouette_arclen"] = sil(X_arclen, labels_arclen)
        except Exception:
            metrics["silhouette_arclen"] = np.nan
    else:
        metrics["silhouette_time"] = np.nan
        metrics["silhouette_arclen"] = np.nan

    metrics["n_clusters_time"] = int(np.unique(labels_time).size)
    metrics["n_clusters_arclen"] = int(np.unique(labels_arclen).size)
    metrics["noise_frac_time"] = float(np.mean(labels_time == -1)) if labels_time.size else np.nan
    metrics["noise_frac_arclen"] = float(np.mean(labels_arclen == -1)) if labels_arclen.size else np.nan
    return metrics


def _plot_histogram_r(r: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r, bins=50, color="#2a9d8f", alpha=0.85, edgecolor="white")
    ax.set_title("Distance Inflation Ratio r = d_time / (d_arclen + eps)")
    ax.set_xlabel("r")
    ax.set_ylabel("Pair count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pca_scatter(
    X_time: np.ndarray,
    labels_time: np.ndarray,
    X_arclen: np.ndarray,
    labels_arclen: np.ndarray,
    out_path: Path,
    rng: np.random.Generator,
) -> None:
    import matplotlib.pyplot as plt

    max_points = 5000
    n = X_time.shape[0]
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        Xt = X_time[idx]
        Xa = X_arclen[idx]
        lt = labels_time[idx]
        la = labels_arclen[idx]
    else:
        Xt, Xa, lt, la = X_time, X_arclen, labels_time, labels_arclen

    pca_t = PCA(n_components=2)
    pca_a = PCA(n_components=2)
    Zt = pca_t.fit_transform(Xt)
    Za = pca_a.fit_transform(Xa)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False, sharey=False)

    def draw(ax, Z, labels, title):
        uniq = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        for idx_label, lab in enumerate(uniq):
            mask = labels == lab
            color = "#b0b0b0" if int(lab) == -1 else cmap(idx_label % 20)
            ax.scatter(Z[mask, 0], Z[mask, 1], s=9, alpha=0.7, color=color, label=str(int(lab)))
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)

    draw(axes[0], Zt, lt, "PCA Scatter: Time-index representation")
    draw(axes[1], Za, la, "PCA Scatter: Arc-length representation")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) <= 12:
            ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_artifacts(
    out_dir: Path,
    traj_ids: list[Any],
    labels_time: np.ndarray,
    labels_arclen: np.ndarray,
    metrics: dict[str, Any],
    X_time: np.ndarray,
    X_arclen: np.ndarray,
    pairs_sample: int,
    eps: float,
    rng: np.random.Generator,
) -> None:
    """Save labels, metrics, and plots for one dataset+L run."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df_t = pd.DataFrame({"traj_id": traj_ids, "label": labels_time.astype(int)})
    df_a = pd.DataFrame({"traj_id": traj_ids, "label": labels_arclen.astype(int)})
    df_t.to_csv(out_dir / "labels_time.csv", index=False)
    df_a.to_csv(out_dir / "labels_arclen.csv", index=False)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    i, j = _sample_pairs(X_time.shape[0], int(pairs_sample), rng)
    if i.size:
        dt = np.linalg.norm(X_time[i] - X_time[j], axis=1)
        ds = np.linalg.norm(X_arclen[i] - X_arclen[j], axis=1)
        r = dt / (ds + float(eps))
        _plot_histogram_r(r, out_dir / "r_histogram.png")

    _plot_pca_scatter(
        X_time=X_time,
        labels_time=labels_time,
        X_arclen=X_arclen,
        labels_arclen=labels_arclen,
        out_path=out_dir / "pca_scatter_time_vs_arclen.png",
        rng=rng,
    )


def _normalize_loaded_trajectories(
    trajectories: list[dict[str, Any]],
    dims: int,
    ctx: RunContext,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    dropped = 0
    for traj in trajectories:
        c = clean_trajectory(traj, dims=dims, ctx=ctx)
        if c is None:
            dropped += 1
            continue
        cleaned.append(c)
    if dropped:
        _log(ctx, f"[warn] dropped {dropped} trajectories after cleaning (too short/invalid)")
    return cleaned


def _load_from_table(path: Path, dims_cfg: int | None, ctx: RunContext) -> tuple[list[dict[str, Any]], int]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path)

    col = _detect_table_columns(df)
    if not col["traj_id"] or not col["x"] or not col["y"]:
        raise ValueError(f"{path}: missing required trajectory columns (traj_id,x,y)")

    dims = int(dims_cfg) if dims_cfg is not None else (3 if col["z"] and df[col["z"]].notna().any() else 2)

    out: list[dict[str, Any]] = []
    for tid, grp in df.groupby(col["traj_id"], sort=True):
        grp_work = grp.copy()

        t_arr: np.ndarray | None = None
        if col["t"]:
            t_arr = _to_float_time(grp_work[col["t"]])
            sort_key = np.argsort(t_arr, kind="mergesort")
            grp_work = grp_work.iloc[sort_key]
            t_arr = t_arr[sort_key]
        else:
            # preserve source order
            t_arr = None

        p_cols = [col["x"], col["y"]]
        if dims == 3:
            if col["z"]:
                p_cols.append(col["z"])
            else:
                # fallback z=0 when requested dims=3 but z missing
                grp_work = grp_work.copy()
                grp_work["__z_fallback__"] = 0.0
                p_cols.append("__z_fallback__")

        p = grp_work[p_cols].to_numpy(dtype=float)
        out.append({"id": tid, "t": t_arr, "p": p})

    return out, dims


def _load_from_npz(path: Path, dims_cfg: int | None) -> tuple[list[dict[str, Any]], int]:
    data = np.load(path, allow_pickle=True)
    if "trajectories" not in data:
        raise ValueError(f"{path}: npz must contain 'trajectories'")

    traj_arr = data["trajectories"]
    times = data["times"] if "times" in data else None

    out: list[dict[str, Any]] = []

    if isinstance(traj_arr, np.ndarray) and traj_arr.ndim == 3:
        N, _, D = traj_arr.shape
        dims = int(dims_cfg) if dims_cfg is not None else (3 if D >= 3 else 2)
        for i in range(N):
            t = None
            if times is not None:
                t = np.asarray(times[i], dtype=float).reshape(-1)
            out.append({"id": i, "t": t, "p": np.asarray(traj_arr[i], dtype=float)[:, :dims]})
        return out, dims

    # Object/ragged fallback
    if traj_arr.dtype == object:
        inferred_dims = 2
        for i, obj in enumerate(traj_arr):
            p = np.asarray(obj, dtype=float)
            if p.ndim != 2 or p.shape[0] < 2:
                continue
            if p.shape[1] >= 3:
                inferred_dims = 3
                break
        dims = int(dims_cfg) if dims_cfg is not None else inferred_dims
        for i, obj in enumerate(traj_arr):
            p = np.asarray(obj, dtype=float)
            if p.ndim != 2:
                continue
            t = None
            if times is not None and len(times) > i:
                t = np.asarray(times[i], dtype=float).reshape(-1)
            out.append({"id": i, "t": t, "p": p[:, :dims]})
        return out, dims

    raise ValueError(f"{path}: unsupported npz trajectories structure")


def _load_from_pickle(path: Path, dims_cfg: int | None) -> tuple[list[dict[str, Any]], int]:
    with path.open("rb") as fh:
        obj = pickle.load(fh)

    out: list[dict[str, Any]] = []
    inferred_dims = 2

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                x = np.asarray(item.get("x"), dtype=float) if item.get("x") is not None else None
                y = np.asarray(item.get("y"), dtype=float) if item.get("y") is not None else None
                if x is None or y is None:
                    continue
                if x.shape[0] != y.shape[0]:
                    continue
                if item.get("z") is not None:
                    z = np.asarray(item.get("z"), dtype=float)
                    if z.shape[0] == x.shape[0]:
                        p = np.column_stack([x, y, z])
                        inferred_dims = 3
                    else:
                        p = np.column_stack([x, y])
                else:
                    p = np.column_stack([x, y])
                t = None
                if item.get("t") is not None:
                    t = np.asarray(item.get("t"), dtype=float)
                out.append({"id": item.get("id", i), "t": t, "p": p})
            else:
                arr = np.asarray(item, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                if arr.shape[1] >= 3:
                    inferred_dims = 3
                out.append({"id": i, "t": None, "p": arr})
    elif isinstance(obj, dict):
        # dict[id] -> array or dict structure
        for key, val in obj.items():
            if isinstance(val, dict) and val.get("x") is not None and val.get("y") is not None:
                x = np.asarray(val["x"], dtype=float)
                y = np.asarray(val["y"], dtype=float)
                if x.shape[0] != y.shape[0]:
                    continue
                if val.get("z") is not None:
                    z = np.asarray(val["z"], dtype=float)
                    if z.shape[0] == x.shape[0]:
                        p = np.column_stack([x, y, z])
                        inferred_dims = 3
                    else:
                        p = np.column_stack([x, y])
                else:
                    p = np.column_stack([x, y])
                t = np.asarray(val["t"], dtype=float) if val.get("t") is not None else None
                out.append({"id": key, "t": t, "p": p})
            else:
                arr = np.asarray(val, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                if arr.shape[1] >= 3:
                    inferred_dims = 3
                out.append({"id": key, "t": None, "p": arr})
    else:
        raise ValueError(f"{path}: unsupported pickle structure")

    dims = int(dims_cfg) if dims_cfg is not None else inferred_dims
    for t in out:
        t["p"] = np.asarray(t["p"], dtype=float)[:, :dims]
    return out, dims


def load_trajectories(path: str | Path, dims_cfg: int | None, ctx: RunContext) -> tuple[list[dict[str, Any]], int]:
    """Load trajectories from CSV/Parquet/NPZ/Pickle into normalized list form."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    ext = p.suffix.lower()
    if ext in {".csv", ".parquet"}:
        trajs, dims = _load_from_table(p, dims_cfg, ctx)
    elif ext == ".npz":
        trajs, dims = _load_from_npz(p, dims_cfg)
    elif ext in {".pkl", ".pickle"}:
        trajs, dims = _load_from_pickle(p, dims_cfg)
    else:
        raise ValueError(f"Unsupported extension '{ext}' for {p}")

    cleaned = _normalize_loaded_trajectories(trajs, dims=dims, ctx=ctx)
    _log(ctx, f"[load] {p} -> trajectories={len(cleaned)} dims={dims}")
    return cleaned, dims


def _swap_preprocessed_root(path: Path) -> Path | None:
    """
    Swap between data/preprocessed and output/preprocessed for convenience.
    Returns None if path does not match either prefix.
    """
    parts = path.parts
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "preprocessed":
        return Path("output") / "preprocessed" / Path(*parts[2:])
    if len(parts) >= 2 and parts[0] == "output" and parts[1] == "preprocessed":
        return Path("data") / "preprocessed" / Path(*parts[2:])
    return None


def _resolve_input_path(raw_path: str | Path, config_dir: Path, ctx: RunContext) -> Path:
    """Resolve input path with fallback rules before loading."""
    p = Path(raw_path)
    candidates: list[Path] = []

    # 1) as provided
    candidates.append(p)
    # 2) relative to config dir
    if not p.is_absolute():
        candidates.append((config_dir / p).resolve())
    # 3) swap data/output preprocessed roots
    for cand in list(candidates):
        if cand.is_absolute():
            try_rel = None
            try:
                # Keep relative-style fallback if under cwd.
                try_rel = cand.relative_to(Path.cwd())
            except Exception:
                pass
            if try_rel is not None:
                swapped = _swap_preprocessed_root(try_rel)
                if swapped is not None:
                    candidates.append(swapped)
                    candidates.append((config_dir / swapped).resolve())
        else:
            swapped = _swap_preprocessed_root(cand)
            if swapped is not None:
                candidates.append(swapped)
                candidates.append((config_dir / swapped).resolve())

    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand.exists():
            if cand != p:
                _log(ctx, f"[info] resolved missing path '{p}' -> '{cand}'")
            return cand

    raise FileNotFoundError(f"Input file not found: {p}")


def _build_representation(
    trajectories: list[dict[str, Any]],
    L: int,
    rep_kind: str,
    dataset_type: str,
) -> tuple[list[Any], np.ndarray, int]:
    ids: list[Any] = []
    points_list: list[np.ndarray] = []
    constant_count = 0

    for traj in trajectories:
        if rep_kind == "time":
            if dataset_type == "raw":
                p = resample_uniform_time(traj, L)
            else:
                # Resampled dataset: keep as-is when length matches L, otherwise align by index.
                p0 = traj["p"]
                p = p0 if p0.shape[0] == L else _resample_uniform_index(traj, L)
        elif rep_kind == "arclen":
            p, is_constant = resample_arclength(traj, L)
            if is_constant:
                constant_count += 1
        else:
            raise ValueError(f"Unknown representation kind: {rep_kind}")

        ids.append(traj["id"])
        points_list.append(p)

    X = vectorize(points_list)
    return ids, X, constant_count


def _process_one_dataset_L(
    dataset_name: str,
    dataset_type: str,
    trajectories: list[dict[str, Any]],
    dims: int,
    L: int,
    cfg: dict[str, Any],
    run_dir: Path,
    ctx: RunContext,
) -> dict[str, Any]:
    _log(ctx, f"[run] dataset={dataset_name} type={dataset_type} L={L} N={len(trajectories)}")

    ids_t, X_time, const_time = _build_representation(trajectories, L=L, rep_kind="time", dataset_type=dataset_type)
    ids_s, X_arc, const_arc = _build_representation(trajectories, L=L, rep_kind="arclen", dataset_type=dataset_type)

    if ids_t != ids_s:
        raise RuntimeError("Trajectory ID alignment mismatch between time and arclength representations.")

    clustering_cfg = cfg.get("clustering", {}) or {}
    random_state = int(clustering_cfg.get("random_state", 11))
    rng = np.random.default_rng(random_state + int(L) + abs(hash(dataset_name)) % 1000)

    labels_t, used_method_t, X_time_fit = cluster(X_time, clustering_cfg, random_state=random_state)
    labels_s, used_method_s, X_arc_fit = cluster(X_arc, clustering_cfg, random_state=random_state)

    allow_sil = str(clustering_cfg.get("method", "")).lower() in {"kmeans", "agglomerative"}

    metrics_cfg = cfg.get("metrics", {}) or {}
    metrics = compute_metrics(
        X_time=X_time,
        X_arclen=X_arc,
        labels_time=labels_t,
        labels_arclen=labels_s,
        pairs_sample=int(metrics_cfg.get("pairs_sample", 50000)),
        eps=float(metrics_cfg.get("eps", 1e-8)),
        rng=rng,
        allow_silhouette=allow_sil,
    )

    metrics.update(
        {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "L": int(L),
            "N": int(X_time.shape[0]),
            "D": int(dims),
            "clustering_method_requested": str(clustering_cfg.get("method", "kmeans")),
            "clustering_method_time": used_method_t,
            "clustering_method_arclen": used_method_s,
            "constant_arclen_trajectories": int(const_arc),
            "constant_time_trajectories": int(const_time),
        }
    )

    ds_dir = run_dir / dataset_type / f"{dataset_name}_L{L}"
    save_artifacts(
        out_dir=ds_dir,
        traj_ids=ids_t,
        labels_time=labels_t,
        labels_arclen=labels_s,
        metrics=metrics,
        X_time=X_time,
        X_arclen=X_arc,
        pairs_sample=int(metrics_cfg.get("pairs_sample", 50000)),
        eps=float(metrics_cfg.get("eps", 1e-8)),
        rng=rng,
    )
    return metrics


def _markdown_table_from_df(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_readme(run_dir: Path, cfg: dict[str, Any], summary_df: pd.DataFrame) -> None:
    view_cols = [
        "dataset_type",
        "dataset_name",
        "L",
        "N",
        "D",
        "clustering_method_time",
        "ARI",
        "r_median",
        "r_p90",
        "silhouette_time",
        "silhouette_arclen",
    ]
    df_view = summary_df[[c for c in view_cols if c in summary_df.columns]].copy()

    text = []
    text.append("# Velocity Leakage Audit")
    text.append("")
    text.append("## Config Used")
    text.append("```yaml")
    text.append(yaml.safe_dump(cfg, sort_keys=False))
    text.append("```")
    text.append("")
    text.append("## Results Summary")
    text.append(_markdown_table_from_df(df_view))
    text.append("")
    text.append("## Interpretation Rules")
    text.append("- ARI >= 0.90: negligible leakage")
    text.append("- 0.70 <= ARI < 0.90: moderate leakage")
    text.append("- ARI < 0.70: strong leakage")
    text.append("- r_median > 1.3 or r_p90 > 2.0: strong distance inflation from velocity leakage")
    (run_dir / "README.md").write_text("\n".join(text), encoding="utf-8")


def _resolve_target_lengths(cfg: dict[str, Any]) -> list[int]:
    sampling = cfg.get("sampling", {}) or {}
    raw_target = sampling.get("raw_target_L", "all")
    base = [40, 60, 70]
    if raw_target == "all":
        return base
    if isinstance(raw_target, int):
        if raw_target not in base:
            raise ValueError("sampling.raw_target_L must be one of 40,60,70 or 'all'")
        return [int(raw_target)]
    if isinstance(raw_target, list):
        vals = sorted({int(v) for v in raw_target})
        for v in vals:
            if v not in base:
                raise ValueError("sampling.raw_target_L list supports only values in {40,60,70}")
        return vals
    raise ValueError("sampling.raw_target_L must be 'all', int, or list[int]")


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "raw_path" not in cfg:
        raise ValueError("Config missing 'raw_path'")
    if "resampled_paths" not in cfg or not isinstance(cfg["resampled_paths"], list):
        raise ValueError("Config missing 'resampled_paths' list")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Velocity leakage audit for trajectory clustering")
    parser.add_argument("--config", required=True, type=Path, help="Path to audit config YAML")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    config_dir = args.config.resolve().parent

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / "velocity_leakage_audit" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(out_dir=run_dir, logs=[])

    _log(ctx, f"[start] velocity leakage audit -> {run_dir}")

    dims_cfg = cfg.get("dims")
    dims_cfg = int(dims_cfg) if dims_cfg is not None else None

    summary_rows: list[dict[str, Any]] = []

    # I) Resampled datasets (40/60/70 etc. as provided)
    for item in cfg.get("resampled_paths", []):
        L = int(item["L"])
        path = _resolve_input_path(item["path"], config_dir=config_dir, ctx=ctx)
        trajs, dims = load_trajectories(path, dims_cfg=dims_cfg, ctx=ctx)
        if len(trajs) < 3:
            _log(ctx, f"[warn] skip resampled L={L} path={path}: only {len(trajs)} valid trajectories")
            continue
        dataset_name = Path(path).stem
        row = _process_one_dataset_L(
            dataset_name=dataset_name,
            dataset_type="resampled",
            trajectories=trajs,
            dims=dims,
            L=L,
            cfg=cfg,
            run_dir=run_dir,
            ctx=ctx,
        )
        summary_rows.append(row)

    # II) Raw dataset over target L(s)
    raw_path = _resolve_input_path(cfg["raw_path"], config_dir=config_dir, ctx=ctx)
    raw_trajs, raw_dims = load_trajectories(raw_path, dims_cfg=dims_cfg, ctx=ctx)
    targets = _resolve_target_lengths(cfg)
    for L in targets:
        if len(raw_trajs) < 3:
            _log(ctx, f"[warn] skip raw L={L}: only {len(raw_trajs)} valid trajectories")
            continue
        row = _process_one_dataset_L(
            dataset_name=Path(raw_path).stem,
            dataset_type="raw",
            trajectories=raw_trajs,
            dims=raw_dims,
            L=L,
            cfg=cfg,
            run_dir=run_dir,
            ctx=ctx,
        )
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No dataset produced valid outputs.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["dataset_type", "dataset_name", "L"]).reset_index(drop=True)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    _write_readme(run_dir, cfg, summary_df)
    (run_dir / "logs.txt").write_text("\n".join(ctx.logs), encoding="utf-8")

    _log(ctx, "[done] audit completed")
    _log(ctx, f"[done] summary: {run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
