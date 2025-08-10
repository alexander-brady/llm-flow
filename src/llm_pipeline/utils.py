from typing import Optional, Dict, Any
import pandas as pd
from OmegaConf import DictConfig, OmegaConf
from jinja2 import Template


def render(template_str, **kwargs) -> str:
    """Render a Jinja2 template with the provided keyword arguments."""
    return Template(template_str).render(**kwargs)


def cfg_to_dict(cfg_node: Optional[DictConfig]) -> Dict[str, Any]:
    """
    Convert a DictConfig object to a regular dictionary.
    """
    return {} if cfg_node is None else OmegaConf.to_container(cfg_node, resolve=True)


def extract_params(params_node: Optional[DictConfig]) -> Dict[str, Dict[str, Any]]:
    """
    Extract all parameters from the configuration.
    
    Args:
        params_node (Optional[DictConfig]): The configuration node containing model parameters.
        
    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing the extracted parameters.
    """
    params_dict = { 'standard': {}, 'guided': {} }
    for key, kwargs in cfg_to_dict(params_node).items():
        params_dict[key] = kwargs
    return params_dict


def build_results(
    df: pd.DataFrame,
    filename: str,
    results: Dict[str, list[str]]
) -> pd.DataFrame:
    """
    Build the final results DataFrame, including all intermediate outputs.
    Args:
        df (pd.DataFrame): The input DataFrame containing the 'content' column.
        filename (str): The name of the output file.
        results (Dict[str, list[str]]): The intermediate results from the model.

    Returns:
        pd.DataFrame: The final results DataFrame.
    """
    return pd.DataFrame({
        'title': df['title'],
        'filename': filename,
        'file_index': df.index,
        **results
    })