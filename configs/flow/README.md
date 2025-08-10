# Flow Configs

Flows describe the steps that will be taken to process user input and generate a response. Each step continues from the previous one, sharing context and information.

## Format

Each flow consists of a series of steps that define the interaction between the user and the assistant. 

**name**

The step's name will be used to identify the output of each step in the final response CSV. 

Each step name must be unique! Otherwise, the outputs will be overwritten.

**type**

The type of the step defines which generation params will be used. These params must be defined in the `model` config.

All params ending in `guided` will use guided decoding to limit output to one of a set of provided choices. These steps must have `choices` defined, which specify the available options for the model to choose from as a list.

If `choices` is set, but no `type`, the default `guided` will be used. If no `type` and no `choices`, the default `standard` will be used.

**system, user, assistant**

Each step can have at most one `system`, `user`, and `assistant` field. These inputs are given to the model in order to generate a response. The content of the `assistant` field will be continued by the model's generation. Steps that only contain a `assistant` field will continue from the previous generation.

Jinja variables will be filled out based on the value of the passed in DataFrame. If `{{ content }}` is used, it will be replaced with the values of the `content` column of the DataFrame.

**Example:**
```yaml
steps:
    - name: QA
      type: standard
      system: >
        You are a helpful assistant.
      user: |
        {{ content }}

    - name: weather
      type: guided
      user: |
        What is the weather?
      assistant: |
        The weather is 
      choices:
        - stormy
        - cloudy
        - sunny
```