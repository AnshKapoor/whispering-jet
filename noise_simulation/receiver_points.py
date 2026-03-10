"""Receiver point mapping helpers for Hannover noise monitoring outputs.

This module maps Doc29 receiver coordinates (UTM32N) to the fixed Hannover
measuring points MP1..MP9. The mapping uses x/y only; z is retained in outputs
but is not used for identification because receiver heights may differ between
tables and generated result files.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


MEASURING_POINTS: List[Dict[str, float | str]] = [
    {"measuring_point": "MP1", "x": 551663.90, "y": 5811763.862},
    {"measuring_point": "MP2", "x": 550989.13, "y": 5813116.422},
    {"measuring_point": "MP3", "x": 553686.61, "y": 5813237.888},
    {"measuring_point": "MP4", "x": 555267.09, "y": 5811988.499},
    {"measuring_point": "MP5", "x": 556553.09, "y": 5813424.585},
    {"measuring_point": "MP6", "x": 542489.95, "y": 5811706.809},
    {"measuring_point": "MP7", "x": 539538.85, "y": 5812485.541},
    {"measuring_point": "MP8", "x": 537135.69, "y": 5813301.049},
    {"measuring_point": "MP9", "x": 542060.92, "y": 5813309.888},
]

_POINT_ORDER = [point["measuring_point"] for point in MEASURING_POINTS]


def annotate_measuring_points(
    df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    point_col: str = "measuring_point",
    max_distance_m: float = 5.0,
) -> pd.DataFrame:
    """Annotate a Doc29 receiver-level frame with MP1..MP9 labels.

    Parameters
    ----------
    df:
        Input DataFrame containing x/y receiver coordinates in UTM32N.
    x_col, y_col:
        Column names for the receiver coordinates.
    point_col:
        Output column name that will hold MP1..MP9 labels.
    max_distance_m:
        Maximum allowed x/y distance from a known measuring point. Rows outside
        this tolerance are labeled as ``UNKNOWN``.
    """
    if df.empty:
        out = df.copy()
        out[point_col] = pd.Series(dtype="object")
        return out

    out = df.copy()
    labels: List[str] = []
    for _, row in out[[x_col, y_col]].iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        best_label = "UNKNOWN"
        best_dist = None
        for point in MEASURING_POINTS:
            dist = ((x - float(point["x"])) ** 2 + (y - float(point["y"])) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_label = str(point["measuring_point"])
        if best_dist is None or best_dist > max_distance_m:
            labels.append("UNKNOWN")
        else:
            labels.append(best_label)

    out[point_col] = labels
    out[point_col] = pd.Categorical(out[point_col], categories=_POINT_ORDER + ["UNKNOWN"], ordered=True)

    front = [point_col]
    rest = [col for col in out.columns if col != point_col]
    return out[front + rest].sort_values([point_col, x_col, y_col]).reset_index(drop=True)
