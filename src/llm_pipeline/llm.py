import pandas as pd
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from vllm import LLM, SamplingParams, RequestOutput
    from transformers import PreTrainedTokenizer


def generate_reasoning_trace(
    df: pd.DataFrame,
    llm: LLM,
    prompt_config: DictConfig,
    tokenizer: PreTrainedTokenizer,
    sampling_params: SamplingParams
) -> List[RequestOutput]:
    """
    Generate a reasoning trace for each item in the DataFrame's 'content' column.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the 'content' column.
        llm (LLM): The language model used for generation.
        prompt_config (DictConfig): The configuration including the prompts.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing the messages.
        sampling_params (SamplingParams): The sampling parameters for generation.
        
    Returns:
        List[RequestOutput]: The generated reasoning traces.
    """
    messages = df['content'].map(lambda x: [
        { 
            'role': 'system',
            'content': prompt_config.system
        },
        {
            'role': 'user',
            'content': x
        }
    ])
    prompts = tokenizer.apply_chat_template(messages.tolist(), tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    return outputs


def classify(
    df: pd.DataFrame,
    llm: LLM,
    prompt_config: DictConfig,
    reasoning: List[RequestOutput],
    tokenizer: PreTrainedTokenizer,
    sampling_params: SamplingParams
) -> List[RequestOutput]:
    """
    Classify the sentiment based on the reasoning outputs.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the 'content' column.
        llm (LLM): The language model used for generation.
        prompt_config (DictConfig): The configuration including the prompts.
        reasoning (List[RequestOutput]): The reasoning outputs from the previous step.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing the messages.
        sampling_params (SamplingParams): The sampling parameters for generation.
        
    Returns:
        List[RequestOutput]: The generated sentiment classifications.
    """
    messages = df['content'].map(lambda x: [
        {
            'role': 'system',
            'content': prompt_config.system
        },
        {
            'role': 'user',
            'content': x
        },
        {
            'role': 'assistant',
            'content': reasoning.outputs[0].text.strip()
        },
        {
            'role': 'user',
            'content': prompt_config.follow_up
        }
    ])
    prompts = tokenizer.apply_chat_template(messages.tolist(), tokenize=False, add_generation_prompt=True)
    prompts = [ prompt + prompt_config.final_answer for prompt in prompts ]

    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    return outputs