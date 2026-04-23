"""Batch runner for compare_experiment_totals across experiment folders.

Runs `noise_simulation/compare_experiment_totals.py` for EXP### in a numeric
range, continues on failures, and writes one log file per experiment.

Inputs:
  - noise_simulation/results/EXP###/summary_mse.csv

Outputs:
  - noise_simulation/results/EXP###/aggregate_totals/*
  - logs/experiments/EXP###_compare_totals.log
  - logs/experiments/compare_totals_batch_summary_EXP###_EXP###.csv

The batch run also attaches the matching all-flights ground truth per
experiment after aggregate totals are created.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunResult:
    experiment: str
    status: str
    return_code: int
    elapsed_sec: float
    summary_csv: str
    out_dir: str
    log_file: str
    error: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exp_name(n: int) -> str:
    return f"EXP{n:03d}"


def _write_log(log_path: Path, lines: List[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _remap_doc29_path(raw_path: str) -> str:
    """Map cloud absolute Doc29 result paths to local repo paths when possible."""

    p = Path(str(raw_path))
    if p.exists():
        return str(p)

    normalized = str(raw_path).replace("\\", "/")
    # Expected cloud path style contains ".../noise_simulation/results/EXP###/..."
    m = re.search(r"(noise_simulation/results/EXP\d{3}/.+)$", normalized)
    if m:
        rel = m.group(1).replace("/", "\\")
        candidate = REPO_ROOT / Path(rel)
        if candidate.exists():
            return str(candidate)
    return str(raw_path)


def _build_localized_summary(exp_dir: Path, summary_csv: Path, log_lines: List[str]) -> tuple[Path, int]:
    """Create a temporary summary with remapped paths if needed.

    Returns:
      (path_to_use, rows_with_both_paths_existing)
    """

    df = pd.read_csv(summary_csv)
    if "subtracks_csv" not in df.columns or "groundtruth_csv" not in df.columns:
        return summary_csv, 0

    original_sub = df["subtracks_csv"].astype(str)
    original_gt = df["groundtruth_csv"].astype(str)
    df["subtracks_csv"] = original_sub.map(_remap_doc29_path)
    df["groundtruth_csv"] = original_gt.map(_remap_doc29_path)

    remapped_sub = int((df["subtracks_csv"] != original_sub).sum())
    remapped_gt = int((df["groundtruth_csv"] != original_gt).sum())
    existing_both = int(
        (
            df["subtracks_csv"].map(lambda x: Path(x).exists())
            & df["groundtruth_csv"].map(lambda x: Path(x).exists())
        ).sum()
    )
    log_lines.append(
        f"Path remap: subtracks={remapped_sub} groundtruth={remapped_gt} rows_with_existing_pairs={existing_both}/{len(df)}"
    )

    tmp_summary = exp_dir / "_summary_mse_local_paths.csv"
    df.to_csv(tmp_summary, index=False)
    return tmp_summary, existing_both


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compare_experiment_totals.py for a range of EXP folders."
    )
    parser.add_argument("--start", type=int, default=1, help="Start experiment number (default: 1).")
    parser.add_argument("--end", type=int, default=62, help="End experiment number (default: 62).")
    parser.add_argument(
        "--results-root",
        default="noise_simulation/results",
        help="Root directory containing EXP### folders.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs/experiments",
        help="Directory for per-experiment logs and batch summary.",
    )
    parser.add_argument(
        "--group-by",
        default="A/D,Runway,aircraft_type",
        help="Passed through to compare_experiment_totals.py --group-by.",
    )
    parser.add_argument(
        "--subtracks-weighting",
        choices=["weighted", "unweighted"],
        default="weighted",
        help="Passed through to compare_experiment_totals.py --subtracks-weighting.",
    )
    parser.add_argument(
        "--tracks-per-cluster",
        type=int,
        default=7,
        help="Passed through to compare_experiment_totals.py --tracks-per-cluster.",
    )
    parser.add_argument(
        "--ground-truth-root",
        default="noise_simulation/results_ground_truth",
        help="Passed through to attach_global_ground_truth_to_experiment_totals.py --ground-truth-root.",
    )
    parser.add_argument(
        "--fallback-ground-truth-csv",
        default="noise_simulation/results_ground_truth/preprocessed_1_final/ground_truth_cumulative.csv",
        help="Fallback all-flights ground truth CSV if the experiment-specific one is missing.",
    )
    args = parser.parse_args()

    if args.start > args.end:
        raise ValueError(f"--start ({args.start}) must be <= --end ({args.end}).")

    results_root = (REPO_ROOT / args.results_root).resolve()
    logs_dir = (REPO_ROOT / args.logs_dir).resolve()
    compare_script = (REPO_ROOT / "noise_simulation" / "compare_experiment_totals.py").resolve()
    attach_script = (REPO_ROOT / "scripts" / "attach_global_ground_truth_to_experiment_totals.py").resolve()
    if not compare_script.exists():
        raise FileNotFoundError(f"Missing script: {compare_script}")
    if not attach_script.exists():
        raise FileNotFoundError(f"Missing script: {attach_script}")

    exp_numbers = list(range(args.start, args.end + 1))
    total_runs = len(exp_numbers)
    results: List[RunResult] = []

    for idx, n in enumerate(exp_numbers, start=1):
        exp = _exp_name(n)
        exp_dir = results_root / exp
        summary_csv = exp_dir / "summary_mse.csv"
        out_dir = exp_dir / "aggregate_totals"
        log_file = logs_dir / f"{exp}_compare_totals.log"

        print(f"[run {idx}/{total_runs}] {exp} ...")
        t0 = time.perf_counter()
        log_lines = [
            f"Experiment: {exp}",
            f"Started (UTC): {_utc_now()}",
            f"Summary CSV: {summary_csv}",
            f"Output dir: {out_dir}",
        ]

        if not exp_dir.exists():
            elapsed = time.perf_counter() - t0
            msg = f"Experiment folder not found: {exp_dir}"
            log_lines.extend([f"Status: FAILED", f"Error: {msg}", f"Elapsed seconds: {elapsed:.2f}"])
            _write_log(log_file, log_lines)
            results.append(
                RunResult(
                    experiment=exp,
                    status="failed",
                    return_code=2,
                    elapsed_sec=elapsed,
                    summary_csv=str(summary_csv),
                    out_dir=str(out_dir),
                    log_file=str(log_file),
                    error=msg,
                )
            )
            print(f"[run {idx}/{total_runs}] {exp} failed (missing folder)")
            continue

        if not summary_csv.exists():
            elapsed = time.perf_counter() - t0
            msg = f"Missing summary_mse.csv: {summary_csv}"
            log_lines.extend([f"Status: FAILED", f"Error: {msg}", f"Elapsed seconds: {elapsed:.2f}"])
            _write_log(log_file, log_lines)
            results.append(
                RunResult(
                    experiment=exp,
                    status="failed",
                    return_code=2,
                    elapsed_sec=elapsed,
                    summary_csv=str(summary_csv),
                    out_dir=str(out_dir),
                    log_file=str(log_file),
                    error=msg,
                )
            )
            print(f"[run {idx}/{total_runs}] {exp} failed (missing summary)")
            continue

        summary_to_use, existing_pairs = _build_localized_summary(exp_dir, summary_csv, log_lines)
        if existing_pairs == 0:
            elapsed = time.perf_counter() - t0
            msg = (
                "No existing subtracks/groundtruth path pairs after local remap. "
                "Copy missing Doc29 result files locally or run comparison on cloud."
            )
            log_lines.extend([f"Status: FAILED", f"Error: {msg}", f"Elapsed seconds: {elapsed:.2f}"])
            _write_log(log_file, log_lines)
            results.append(
                RunResult(
                    experiment=exp,
                    status="failed",
                    return_code=3,
                    elapsed_sec=elapsed,
                    summary_csv=str(summary_csv),
                    out_dir=str(out_dir),
                    log_file=str(log_file),
                    error=msg,
                )
            )
            print(f"[run {idx}/{total_runs}] {exp} failed (no local path pairs)")
            continue

        cmd = [
            sys.executable,
            str(compare_script),
            "--summary",
            str(summary_to_use),
            "--out",
            str(out_dir),
            "--group-by",
            args.group_by,
            "--subtracks-weighting",
            args.subtracks_weighting,
            "--tracks-per-cluster",
            str(args.tracks_per_cluster),
        ]
        log_lines.append("Command:")
        log_lines.append(" ".join(cmd))

        compare_proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        attach_proc = None
        if compare_proc.returncode == 0:
            attach_cmd = [
                sys.executable,
                str(attach_script),
                "--results-root",
                args.results_root,
                "--ground-truth-root",
                args.ground_truth_root,
                "--fallback-ground-truth-csv",
                args.fallback_ground_truth_csv,
                "--start",
                str(n),
                "--end",
                str(n),
            ]
            log_lines.append("Attach command:")
            log_lines.append(" ".join(attach_cmd))
            attach_proc = subprocess.run(
                attach_cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
            )

        elapsed = time.perf_counter() - t0
        ok = compare_proc.returncode == 0 and (attach_proc is None or attach_proc.returncode == 0)
        status = "ok" if ok else "failed"
        if compare_proc.returncode != 0:
            err = compare_proc.stderr.strip() or "compare_experiment_totals.py returned non-zero exit code"
        elif attach_proc is not None and attach_proc.returncode != 0:
            err = attach_proc.stderr.strip() or "attach_global_ground_truth_to_experiment_totals.py returned non-zero exit code"
        else:
            err = ""

        log_lines.append(f"Status: {'OK' if ok else 'FAILED'}")
        log_lines.append(f"Compare return code: {compare_proc.returncode}")
        if attach_proc is not None:
            log_lines.append(f"Attach return code: {attach_proc.returncode}")
        log_lines.append(f"Elapsed seconds: {elapsed:.2f}")
        if compare_proc.stdout:
            log_lines.append("COMPARE STDOUT:")
            log_lines.append(compare_proc.stdout.rstrip())
        if compare_proc.stderr:
            log_lines.append("COMPARE STDERR:")
            log_lines.append(compare_proc.stderr.rstrip())
        if attach_proc is not None and attach_proc.stdout:
            log_lines.append("ATTACH STDOUT:")
            log_lines.append(attach_proc.stdout.rstrip())
        if attach_proc is not None and attach_proc.stderr:
            log_lines.append("ATTACH STDERR:")
            log_lines.append(attach_proc.stderr.rstrip())
        _write_log(log_file, log_lines)

        results.append(
            RunResult(
                experiment=exp,
                status=status,
                return_code=0 if ok else (
                    attach_proc.returncode
                    if attach_proc is not None and attach_proc.returncode != 0
                    else compare_proc.returncode
                ),
                elapsed_sec=elapsed,
                summary_csv=str(summary_csv),
                out_dir=str(out_dir),
                log_file=str(log_file),
                error=err,
            )
        )
        print(f"[run {idx}/{total_runs}] {exp} {status} in {elapsed:.1f}s")

    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_file = logs_dir / f"compare_totals_batch_summary_{_exp_name(args.start)}_{_exp_name(args.end)}.csv"
    with summary_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment",
                "status",
                "return_code",
                "elapsed_sec",
                "summary_csv",
                "out_dir",
                "log_file",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "experiment": r.experiment,
                    "status": r.status,
                    "return_code": r.return_code,
                    "elapsed_sec": f"{r.elapsed_sec:.3f}",
                    "summary_csv": r.summary_csv,
                    "out_dir": r.out_dir,
                    "log_file": r.log_file,
                    "error": r.error,
                }
            )

    n_ok = sum(1 for r in results if r.status == "ok")
    n_failed = len(results) - n_ok
    print(f"Done. ok={n_ok} failed={n_failed}")
    print(f"Batch summary: {summary_file}")


if __name__ == "__main__":
    main()
