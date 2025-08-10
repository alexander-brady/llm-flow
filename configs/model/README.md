# Model Configs

Model Configs define the parameters and settings for the model to use in the flow pipeline. Uses local vLLM instance as backend.

## Format

**name**

The huggingface model name, e.g. `openai/gpt-oss-120b`.

**tokenizer**

Optional: The tokenizer to use for text splitting and encoding. If not specified, the tokenizer will default to the model's name.

**params**

This dictionary contains the parameters for each unique prompt step, including temperature, top_p, max_length, and repetition_penalty. 

Recommended to include `standard` and `guided` configurations. Further configurations can be added as needed. Any configuration not explicitly defined will fall back to the `standard` settings. Configurations ending in `guided` will use guided decoding to limit output to one of a set of provided choices.

**Example**
```yaml
name: my_model
tokenizer: my_tokenizer

params:
  standard:
    temperature: 0.7
    top_p: 0.9
    max_length: 1024
    repetition_penalty: 1.0

  guided:
    temperature: 0.0
    max_length: 512
```