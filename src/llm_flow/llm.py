from typing import Tuple, Dict, List
from logging import Logger

import pandas as pd
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from vllm.sampling_params import GuidedDecodingParams

from .utils import render


def init_models(model_cfg: DictConfig) -> Tuple[LLM, PreTrainedTokenizer]:
    """
    Initialize the language model based on the configuration.

    Args:
        model_cfg (DictConfig): The configuration for the model.

    Returns:
        Tuple[LLM, PreTrainedTokenizer]: The initialized language model and tokenizer.
    """
    llm = LLM(model=model_cfg.name)
    tok_name = model_cfg.get('tokenizer', model_cfg.name)
    tokenizer = PreTrainedTokenizer.from_pretrained(tok_name)
    return llm, tokenizer


def build_step_params(steps_cfg: list[DictConfig], params: Dict[str, dict]) -> Dict[str, SamplingParams]:
    """
    Build the sampling parameters for each flow step.

    Args:
        steps_cfg (list[DictConfig]): The configuration for the flow steps.

    Returns:
        Dict[str, SamplingParams]: The sampling parameters for each flow step.
    """
    out = {}
    for step in steps_cfg:
        step_type = (step.get('type') or "").strip()
        if step_type.endswith("guided") or (not step_type and step.get('choices')):
            out[step.name] = SamplingParams(
                **params.get("guided", {}),
                guided_decoding=GuidedDecodingParams(choice=step.choices),
            )
        else:
            out[step.name] = SamplingParams(**params.get("standard", {}))
    return out


def run_steps_on_df(
    df: pd.DataFrame,
    steps_cfg: list[DictConfig],
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    step_params: Dict[str, SamplingParams],
    *,
    log: Logger
) -> Dict[str, List[str]]:
    """
    Execute the flow steps on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        steps_cfg (list[DictConfig]): The configuration for the flow steps.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text splitting and encoding.
        llm (vllm.LLM): The language model to use for text generation.
        step_params (Dict[str, SamplingParams]): The parameters for each flow step.
        log (Logger): The logger to use for logging errors.

    Returns:
        A dictionary containing the results of each flow step.
    """
    results, messages, prev = {}, [], None
    for step in steps_cfg:
        try:
            messages = extend_prompts(messages, step, prev, df)
            prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate(prompts=prompts, sampling_params=step_params[step.name])
            prev = [o.outputs[0].text.strip() for o in outputs]
            results[step.name] = prev
        except Exception as e:
            log.exception("Failed on %s for step %s: %s", getattr(df, "name", "<df>"), step.name, e)
    return results


def extend_prompts(
    messages: List[List[Dict[str, str]]],
    step: DictConfig,
    prev: List[RequestOutput] | None,
    df: pd.DataFrame
) -> List[List[Dict[str, str]]]:
    """
    Extend the prompts for the current step based on previous messages and the DataFrame.

    Args:
        messages (List[List[Dict[str, str]]]): The list of previous messages.
        step (DictConfig): The configuration for the current step.
        prev (List[RequestOutput] | None): The list of previous outputs.
        df (pd.DataFrame): The DataFrame being processed.

    Returns:
        A tuple containing the updated messages, and tokenized prompts.
    """    
    if prev:
        for lst, output in zip(messages, prev):
            lst.append({
                "role": "assistant",
                "content": output[0].text.strip(),
            })

    if step.get('system'):
        for lst, (_, row) in zip(messages, df.iterrows()):
            lst.append({
                "role": "system",
                "content": render(step.system, **row.to_dict()),
            })
        
    if step.get('user'):
        for lst, (_, row) in zip(messages, df.iterrows()):
            lst.append({
                "role": "user",
                "content": render(step.user, **row.to_dict()),
            })
            
    if step.get('assistant'):
        if messages and messages[0][-1]['role'] == 'assistant':
            for lst, (_, row) in zip(messages, df.iterrows()):
                lst[-1]['content'] += render(step.assistant, **row.to_dict())
        else:
            for lst, (_, row) in zip(messages, df.iterrows()):
                lst.append({
                    "role": "assistant",
                    "content": render(step.assistant, **row.to_dict()),
                })

    return messages