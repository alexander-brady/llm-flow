from logging import Logger

import pandas as pd
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .utils import render


def init_models(model_cfg: DictConfig) -> tuple[LLM, PreTrainedTokenizer]:
    """
    Initialize the language model based on the configuration.

    Args:
        model_cfg (DictConfig): The configuration for the model.

    Returns:
        Tuple[LLM, PreTrainedTokenizer]: The initialized language model and tokenizer.
    """
    llm = LLM(model=model_cfg.name)
    tok_name = model_cfg.get("tokenizer", model_cfg.name)
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return llm, tokenizer


def build_step_params(steps_cfg: list[DictConfig], params: dict[str, dict]) -> dict[str, SamplingParams]:
    """
    Build the sampling parameters for each flow step.

    Args:
        steps_cfg (list[DictConfig]): The configuration for the flow steps.

    Returns:
        Dict[str, SamplingParams]: The sampling parameters for each flow step.
    """
    out = {}
    for step in steps_cfg:
        step_type = (step.get("type") or "").strip()
        if step_type.endswith("guided") or (not step_type and step.get("choices")):
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
    step_params: dict[str, SamplingParams],
    *,
    log: Logger,
) -> dict[str, list[str]]:
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
    results, outputs = {}, None
    messages: list[list[dict[str, str]]] = [[] for _ in range(len(df))]
    for step in steps_cfg:
        try:
            messages = extend_prompts(messages, step, outputs, df)
            prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            request = llm.generate(prompts=prompts, sampling_params=step_params[step.name])
            outputs = [r.outputs[0].text.strip() for r in request]
            results[step.name] = outputs
        except Exception as e:
            log.exception("Failed on %s for step %s: %s", getattr(df, "name", "<df>"), step.name, e)
    return results


def extend_prompts(
    messages: list[list[dict[str, str]]], step: DictConfig, prev_output: list[str] | None, df: pd.DataFrame
) -> list[list[dict[str, str]]]:
    """
    Extend the prompts for the current step based on previous messages and the DataFrame.

    Args:
        messages (List[List[Dict[str, str]]]): The list of previous messages.
        step (DictConfig): The configuration for the current step.
        prev_output (List[str] | None): The list of previous generated text outputs.
        df (pd.DataFrame): The DataFrame being processed.

    Returns:
        A tuple containing the updated messages, and tokenized prompts.
    """
    if prev_output:
        for lst, gen in zip(messages, prev_output, strict=True):
            lst.append(
                {
                    "role": "assistant",
                    "content": gen,
                }
            )

    if step.get("system"):
        for lst, (_, row) in zip(messages, df.iterrows(), strict=True):
            lst.append(
                {
                    "role": "system",
                    "content": render(step.system, **row.to_dict()),
                }
            )

    if step.get("user"):
        for lst, (_, row) in zip(messages, df.iterrows(), strict=True):
            lst.append(
                {
                    "role": "user",
                    "content": render(step.user, **row.to_dict()),
                }
            )

    if step.get("assistant"):
        if messages and messages[0][-1]["role"] == "assistant":
            for lst, (_, row) in zip(messages, df.iterrows(), strict=True):
                lst[-1]["content"] += render(step.assistant, **row.to_dict())
        else:
            for lst, (_, row) in zip(messages, df.iterrows(), strict=True):
                lst.append(
                    {
                        "role": "assistant",
                        "content": render(step.assistant, **row.to_dict()),
                    }
                )

    return messages
