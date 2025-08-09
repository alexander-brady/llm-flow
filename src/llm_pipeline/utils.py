from typing import List, Optional, Dict, Any
import pandas as pd
from vllm import RequestOutput
from OmegaConf import DictConfig, OmegaConf


def build_results(
    df: pd.DataFrame,
    filename: str,
    reasoning: List[RequestOutput],
    classification: List[RequestOutput]
) -> pd.DataFrame:
    """
    Build the final results DataFrame from the input DataFrame and the reasoning and classification outputs.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'content' column.
        reasoning (List[RequestOutput]): The reasoning outputs from the previous step.
        classification (List[RequestOutput]): The classification outputs from the previous step.

    Returns:
        pd.DataFrame: The final results DataFrame.
    """
    df = pd.DataFrame({
        'title': df['title'],
        'filename': filename,
        'file_index': df.index,
        'reasoning': [r.outputs[0].text.strip() for r in reasoning],
        'classification': [c.outputs[0].text.strip() for c in classification],
        'date': df['publishedAt']
    })

def cfg_to_dict(cfg_node: Optional[DictConfig]) -> Dict[str, Any]:
    """
    Convert a DictConfig object to a regular dictionary.
    """
    return {} if cfg_node is None else OmegaConf.to_container(cfg_node, resolve=True)