from pathlib import Path
import hydra
from omegaconf import DictConfig

from .logging_setup import init_logging
from .pipeline import run_pipeline
from .io import save_results


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    log = init_logging(cfg)
    df = run_pipeline(cfg, log=log)
    out_csv = Path(".") / f"{cfg.output_name}.csv"
    save_results(df, out_csv)
    log.info("Saved %d rows -> %s", len(df), out_csv.resolve())


if __name__ == "__main__":
    main() # type: ignore