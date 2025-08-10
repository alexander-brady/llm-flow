# LLM Flow

Lightweight framework for building multi-step pipelines, called [flows](configs/flow/README.md), powered by large language models. Each step can answer prompts, classify from predefined options, or build on previous results to create complex multi-step tasks. Designed for simplicity and scalability, LLM Flow makes it easy to prototype, iterate, and deploy simple yet intelligent workflows on large datasets.

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

This project uses **[Hydra](https://hydra.cc)** for configuration management. These configurations define not just the model and data parameters, but also the entire pipeline behavior.

* Default configs live in the [`configs/`](configs) directory.
* You can override any parameter from the command line.
* For more details, see the documentation in the respective `configs` directories.

Example:

```bash
python -m src.llm_flow \
    model.name=openai/gpt-oss-120b \
    model.temperature=0.5 \
    data_dir=data/processed \
    flow=default
```

## Designing Flows

> For detailed documentation, see [configs/flow](configs/flow/README.md).

Flows define multi-step interactions, written in simple YAML.  Each step builds on previous outputs, allowing you to design clear, structured LLM workflows.

Flows are written in the `configs/flow` folder. You can override the current prompt flow run via config or Hydra CLI.

## Data Format

> For our original experiments, we used news articles from the [GNews API](http://gnews.io).

Place your data in the `data/` directory (or override `data_dir` via config/CLI). Data must be Parquet or CSV format. Each file within the directory will be treated as part of the dataset.

## Quickstart

> See [run.sh](run.sh) for an example SLURM job script, designed for the [ETH Zurich Euler cluster](https://ele.ethz.ch/resources-and-infrastructure/infrastructure/computational-ressources.html).

1. Prepare your data in `data/`
2. Adjust parameters and prompts in [`configs/config.yaml`](configs/config.yaml) if needed.
3. Run:

   ```bash
   python -m src.llm_flow
   ```
4. Results are saved automatically to the Hydra run directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`).


## Outputs

Each run produces:

* `results.csv` — classification results with reasoning traces
* Hydra config snapshot (`.hydra/`) for reproducibility
* Optional logs in the run directory

The `results.csv` file has the following format:
- `filename`: Name of the file the item in this row is from
- `file_index`: Index of the item inside `filename`
- `<step_name>`: Output of the LLM for this step
