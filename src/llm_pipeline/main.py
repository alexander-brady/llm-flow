import logging
from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from tqdm import tqdm

from .utils import cfg_to_dict, build_results
from .io import load_articles_from_dir
from .llm import generate_reasoning_trace, classify
from .logging_setup import setup_logging   

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    setup_logging(cfg)
    log = logging.getLogger(__name__)

    # --- Model + tokenizer ---
    llm = LLM(model=cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer or cfg.model.name)

    # --- Sampling params ---
    generation_params = SamplingParams(**cfg_to_dict(cfg.generation_params))
    guided_generation_params = SamplingParams(
        **cfg_to_dict(cfg.guided_generation_params),
        guided_decoding=GuidedDecodingParams(choice=cfg.prompts.choices),
    )

    # --- Iterate files ---
    data_dir = Path(cfg.data_dir)
    files = sorted([p for p in data_dir.iterdir() if p.is_file()])
    final_outputs = pd.DataFrame()

    log.info("Processing %d files from %s", len(files), data_dir)

    for filename, df in tqdm(load_articles_from_dir(data_dir), total=len(files)):
        try:
            reasoning = generate_reasoning_trace(
                df, llm, cfg.prompts, tokenizer, generation_params
            )
            classifications = classify(
                df, llm, cfg.prompts, reasoning, tokenizer, guided_generation_params
            )
            results_df = build_results(df, filename, reasoning, classifications)
            final_outputs = pd.concat([final_outputs, results_df], ignore_index=True)
        except Exception as e:
            log.exception("Failed on %s: %s", filename, e)

    # Persist to Hydra run dir (unique per run)
    out_csv = Path(cfg.run.dir) /  f'{cfg.csv_name}.csv'
    final_outputs.to_csv(out_csv, index=False)
    log.info("Saved %d rows -> %s", len(final_outputs), out_csv.resolve())

    return final_outputs

if __name__ == "__main__":
    main()