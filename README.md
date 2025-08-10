# LLM-Powered Classification

This project uses the reasoning capabilities of **large language models (LLMs)** to perform classification tasks. Using step-based prompting, we can guide the model to classify items based on their content and context.

## Installation

1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Download tabular data files to the `data/` directory. Data must be Parquet or CSV format. 

## Configuration

This project uses **[Hydra](https://hydra.cc)** for configuration management. These configurations define not just the model and data parameters, but also the entire pipeline behavior (see [configs/prompts](configs/prompts/README.md)).

* Default configs live in the [`configs/`](configs) directory.
* You can override any parameter from the command line.
* For more details, see the documentation in the respective `configs` directories.

Example:

```bash
python -m src.llm_pipeline \
    model.name=openai/gpt-oss-120b \
    model.temperature=0.5 \
    data_dir=data/processed
```


## Data Format

> For our experiments, we used news articles from the [GNews API](http://gnews.io).

Place your data in the `data/` directory (or override `data_dir` via config/CLI). Data must be Parquet or CSV format. Each file within the directory will be treated as part of the dataset.

## Quickstart

> See [run.sh](run.sh) for an example SLURM job script, designed for the [ETH Zurich Euler cluster](https://ele.ethz.ch/resources-and-infrastructure/infrastructure/computational-ressources.html).

1. Prepare your data in `data/`
2. Adjust parameters and prompts in [`configs/config.yaml`](configs/config.yaml) if needed.
3. Run:

   ```bash
   python -m src.llm_pipeline
   ```
4. Results are saved automatically to the Hydra run directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`).


## Outputs

Each run produces:

* `final_outputs.csv` — classification results with reasoning traces
* Hydra config snapshot (`.hydra/`) for reproducibility
* Optional logs in the run directory

The `final_outputs.csv` file has the following format:
- `filename`: Name of the file the item in this row is from
- `file_index`: Index of the item inside `filename`
- `<step_name>`: Output of the LLM for this step