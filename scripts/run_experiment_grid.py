"""Run a grid of clustering experiments defined in a YAML file."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import run_experiment


def _is_experiment_complete(exp_dir: Path) -> bool:
    """Return True if key outputs for an experiment exist."""
    if not exp_dir.exists():
        return False
    required = [
        exp_dir / "experiment_log.txt",
        exp_dir / "metrics_global.csv",
        exp_dir / "metrics_by_flow.csv",
        exp_dir / "labels_ALL.csv",
    ]
    return all(p.exists() for p in required)


def iter_param_grid(param_dict: dict) -> list[dict]:
    keys = []
    values = []
    for key, val in param_dict.items():
        if isinstance(val, list):
            keys.append(key)
            values.append(val)
    if not keys:
        return [param_dict]
    combos = []
    for product_vals in itertools.product(*values):
        cfg = copy.deepcopy(param_dict)
        for k, v in zip(keys, product_vals):
            cfg[k] = v
        combos.append(cfg)
    return combos


def main(grid_path: Path) -> None:
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8")) or {}
    experiments = grid.get("experiments", [])
    base_output = grid.get("output", {}).get("dir", "output")
    skip_completed = bool(grid.get("output", {}).get("skip_completed", True))
    base_cfg_path = Path("config/backbone_full.yaml")
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    total_runs = 0
    for exp in experiments:
        method = exp["method"]
        method_params = exp.get(method, {})
        if method_params:
            total_runs += len(iter_param_grid(method_params))
        else:
            total_runs += 1
    print(
        f"[grid] loaded {len(experiments)} experiment specs -> {total_runs} planned runs "
        f"(skip_completed={skip_completed})",
        flush=True,
    )

    run_counter = 0
    for exp in experiments:
        exp_name = exp["name"]
        method = exp["method"]
        distance_metric = exp.get("distance_metric", "euclidean")
        distance_params = copy.deepcopy(exp.get("distance_params", {}))

        # Prepare parameter sweeps
        param_space = []
        method_params = exp.get(method, {})
        if method_params:
            param_space = iter_param_grid(method_params)
        else:
            param_space = [{}]

        explicit_name = exp.get("experiment_name")
        if len(param_space) > 1 and not explicit_name:
            raise ValueError(
                f"Experiment '{exp_name}' expands to multiple runs but has no explicit experiment_name. "
                "Provide explicit experiment_name entries for each run to keep numbering stable."
            )

        for idx, params in enumerate(param_space, start=1):
            run_counter += 1
            cfg = copy.deepcopy(base_cfg)
            cfg["clustering"]["method"] = method
            cfg["clustering"]["distance_metric"] = distance_metric
            cfg["clustering"][method] = params
            cfg["clustering"]["distance_params"] = copy.deepcopy(distance_params)
            if "sample_for_fit" in exp:
                cfg["clustering"]["sample_for_fit"] = copy.deepcopy(exp["sample_for_fit"])
            grid_flows = grid.get("flows")
            if grid_flows:
                cfg["flows"] = copy.deepcopy(grid_flows)
            grid_features = grid.get("features")
            if grid_features:
                cfg["features"] = copy.deepcopy(grid_features)
            exp_features = exp.get("features")
            if exp_features:
                cfg.setdefault("features", {})
                cfg["features"].update(copy.deepcopy(exp_features))
            exp_input = exp.get("input")
            if exp_input:
                cfg.setdefault("input", {}).update(copy.deepcopy(exp_input))
            if explicit_name:
                cfg["output"]["experiment_name"] = explicit_name
            else:
                cfg["output"]["experiment_name"] = exp_name
            # Use provided preprocessed CSV from grid only when experiment doesn't override it.
            if "input" in grid and "preprocessed_csv" in grid["input"]:
                cfg.setdefault("input", {})
                if "preprocessed_csv" not in cfg["input"]:
                    cfg["input"]["preprocessed_csv"] = grid["input"]["preprocessed_csv"]

            # Write temp config and run
            tmp_cfg_path = Path(f".tmp_grid_cfg_{exp_name}_{idx}.yaml")
            tmp_cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
            try:
                preprocessed_csv = cfg.get("input", {}).get("preprocessed_csv", "<auto>")
                print(
                    f"[run {run_counter}/{total_runs}] start {cfg['output']['experiment_name']} "
                    f"method={method} distance={distance_metric} input={preprocessed_csv}",
                    flush=True,
                )
                exp_dir = Path(base_output) / "experiments" / cfg["output"]["experiment_name"]
                if skip_completed and _is_experiment_complete(exp_dir):
                    print(f"[run {run_counter}/{total_runs}] skip {cfg['output']['experiment_name']} already complete.", flush=True)
                    continue
                start = time.perf_counter()
                run_experiment(tmp_cfg_path)
                elapsed = time.perf_counter() - start
                print(
                    f"[run {run_counter}/{total_runs}] done {cfg['output']['experiment_name']} in {elapsed:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[run {run_counter}/{total_runs}] error {cfg['output']['experiment_name']} -> {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                tmp_cfg_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a grid of clustering experiments.")
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("experiments/experiment_grid.yaml"),
        help="Path to the experiment grid YAML.",
    )
    args = parser.parse_args()
    main(args.grid)
