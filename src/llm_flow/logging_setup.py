import sys
import logging
from omegaconf import DictConfig


def setup_logging(cfg: DictConfig):
    """
    Initialize global logging configuration.

    Args:
        cfg (DictConfig): The configuration object.
    """
    logging.basicConfig(
        level=cfg.get("logging", {}).get("level", logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def init_logging(cfg: DictConfig) -> logging.Logger:
    """
    Initialize logging for the application and return the logger.
    
    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        logging.Logger: The initialized logger.
    """
    setup_logging(cfg)
    return logging.getLogger(__name__)