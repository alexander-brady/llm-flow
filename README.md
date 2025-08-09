# LLM-Powered Classification

This repository uses the reasoning capabilities of large language models (LLMs) for classification tasks. By leveraging the contextual understanding and inference abilities of LLMs, we aim to improve the accuracy and efficiency of classification processes across various domains.

## Installation

To install the necessary dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Configuration

This project uses the Hydra configuration system. You can modify the configuration files located in the `configs` directory to customize the behavior of the LLM. When you run the main file, you can override specific configuration options using the command line.

Example:

```bash
python -m src/llm_pipeline/main.py model.name=gpt-4o-mini model.temperature=0.5 data_dir=data/processed
```

## Data

The data used for training and evaluation is located in the `data` directory. The data must be stored in the `data` folder. Make sure to update the directory in `configs/config.yaml`. The data should be saved as parquet, and must contain the following 3 columns:

1. `title`: The title of the document.
2. `content`: The content of the document.
3. `publishedAt`: The publication date of the document.
