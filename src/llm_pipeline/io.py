import os
from typing import Iterator, Tuple
import pandas as pd

def load_articles_from_dir(dir_path: str) -> Iterator[Tuple[str, pd.DataFrame]]:
    """
    Load downloaded articles from the parquet files in the specified directory.

    Args:
        dir_path (str): Folder within the directory where the articles are stored.

    Returns:
        Iterator[Tuple]: An iterator that yields each file in the directory as a tuple of (filename, article data).
    """
    for filename in os.listdir(dir_path):
        if filename.endswith('.parquet'):
            file_path = os.path.join(dir_path, filename)
            filename = filename[:-len('.parquet')]
            yield filename, pd.read_parquet(file_path)