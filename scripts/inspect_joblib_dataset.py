"""Entry-point script for inspecting the April 2022 joblib dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from src.utils.joblib_visualization import load_joblib_dataset


def _format_collection_preview(values: Iterable[Any], max_items: int = 5) -> str:
    """Return a readable preview string for a generic iterable collection.

    Parameters
    ----------
    values:
        Iterable whose items should be included in the preview.
    max_items:
        Maximum number of elements to display before truncating.

    Returns
    -------
    str
        Comma-separated preview of the leading entries.
    """

    # Convert the iterable into a list so slicing can be applied consistently.
    items: list[Any] = list(values)
    preview_items: list[Any] = items[:max_items]
    has_more: bool = len(items) > max_items
    preview: str = ", ".join(str(item) for item in preview_items)
    if has_more:
        # Highlight to the user that only the first few entries are displayed.
        preview += " ..."
    return preview


def format_dataset_preview(dataset: Any) -> str:
    """Create a human-readable summary of the supplied joblib dataset.

    Parameters
    ----------
    dataset:
        Python object produced by :func:`load_joblib_dataset`.

    Returns
    -------
    str
        Textual description that can be printed to the console.
    """

    # DataFrames are displayed using their first few rows, which commonly contain
    # fields such as timestamp, latitude, and longitude.
    if isinstance(dataset, pd.DataFrame):
        return dataset.head().to_string(index=False)

    # Mapping-based datasets allow us to describe the top-level keys.
    if isinstance(dataset, Mapping):
        keys: list[Any] = list(dataset.keys())
        key_preview: str = _format_collection_preview(keys)
        total_entries: int = len(keys)
        return (
            "Mapping dataset with keys: "
            f"{key_preview} (total entries: {total_entries})"
        )

    # Some datasets expose an attribute that holds the underlying DataFrame.
    data_attr: Any = getattr(dataset, "data", None)
    if isinstance(data_attr, pd.DataFrame):
        return data_attr.head().to_string(index=False)

    # Fallback to reporting the type, which still gives users useful context.
    return f"Loaded object of type {type(dataset).__name__}."


def main() -> None:
    """Load the April 2022 joblib dataset and print a concise preview.

    This helper demonstrates how to reuse :func:`load_joblib_dataset` for a
    specific monthly archive that lives one directory above the repository.
    """

    # Derive the path to the joblib archive, which is stored one directory level
    # above the project root according to the user's instructions.
    project_root: Path = Path(__file__).resolve().parent
    joblib_path: Path = project_root.parent / "data_2022_april.joblib"

    # Load the dataset using the shared utility function so the logic stays DRY.
    dataset: Any = load_joblib_dataset(str(joblib_path))

    # Generate a human-friendly summary of what has been loaded.
    preview: str = format_dataset_preview(dataset)

    # Provide the user with both the source path and the preview details.
    print(f"Loaded dataset from {joblib_path}")
    print(preview)


if __name__ == "__main__":
    main()
