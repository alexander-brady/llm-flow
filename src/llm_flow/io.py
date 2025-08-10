from pathlib import Path
from typing import Iterator, Tuple
import pandas as pd


def load_articles_from_dir(dir_path: Path) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Load downloaded articles from the parquet files in the specified directory.

    Args:
        dir_path (Path): Folder within the directory where the articles are stored.

    Returns:
        Iterator[Tuple]: An iterator that yields each file in the directory as a tuple of (filename, article data).
    """
    for filename in dir_path.glob("*"):
        if filename.suffix == ".parquet":
            yield filename.stem, pd.read_parquet(filename)
        elif filename.suffix == ".csv":
            yield filename.stem, pd.read_csv(filename)


def iter_parquet_dir(data_dir: Path) -> Tuple[int, Iterator[Tuple[str, pd.DataFrame]]]:
    """
    Iterate over the parquet files in the specified directory.

    Args:
        data_dir (Path): The directory to search for parquet files.

    Returns:
        Tuple[int, Iterator[Tuple[str, pd.DataFrame]]]: 
            A tuple containing the number of files found and an iterator over the articles.
    """
    file_count = sum(1 for f in data_dir.glob("*") if f.is_file() and f.suffix.lower() in {".parquet", ".csv"})
    return file_count, load_articles_from_dir(data_dir)


def save_results(final_outputs: pd.DataFrame, out_csv: Path):
    """Save the final outputs to a CSV file."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_outputs.to_csv(out_csv, index=False)