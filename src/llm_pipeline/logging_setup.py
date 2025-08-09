import sys
import logging
from omegaconf import DictConfig


def setup_logging(cfg: DictConfig):
    """
    Initialize global logging configuration.
    """
    logging.basicConfig(
        level=cfg.logging.level or logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )