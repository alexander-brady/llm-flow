from pathlib import Path

import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig

from .io import iter_parquet_dir
from .llm import build_step_params, init_models, run_steps_on_df
from .utils import extract_params, build_results


def run_pipeline(cfg: DictConfig, *, log) -> pd.DataFrame:
    # Initialize models and flows
    llm, tokenizer = init_models(cfg.model)
    params = extract_params(cfg.model.params)
    log.info("Loaded model params: %s", params)
    step_params = build_step_params(cfg.flow.steps, params)

    # Load data
    data_dir = Path(cfg.data_dir)
    file_count, file_iter = iter_parquet_dir(data_dir)
    if file_count == 0:
        log.warning("No parquet files found in %s", data_dir)
        return pd.DataFrame()


    # Run flow on each file.
    log.info("Processing %d files from %s", file_count, data_dir)
    final = []
    for filename, df in tqdm(file_iter, total=file_count, desc="Processing files"):
        res = run_steps_on_df(df, cfg.flow.steps, tokenizer, llm, step_params, log=log)
        final.append(build_results(df, filename, res))
        
    return pd.concat(final, ignore_index=True) if final else pd.DataFrame()