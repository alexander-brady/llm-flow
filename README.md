# LLM-Powered Classification

This project uses the reasoning capabilities of **large language models (LLMs)** to perform classification tasks. By leveraging contextual understanding and multi-step inference, it aims to improve classification accuracy and adaptability across domains.

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

---

## ⚙️ Configuration

This project uses **[Hydra](https://hydra.cc)** for configuration management.

* Default configs live in the [`configs/`](configs) directory.
* You can override any parameter from the command line.

Example:

```bash
python -m src.llm_pipeline.main \
    model.name=openai/gpt-oss-120b \
    model.temperature=0.5 \
    data_dir=data/processed
```

---

## 📂 Data Format

Place your data in the `data/` directory (or override `data_dir` via config/CLI).
Data must be **Parquet format** and contain:

| Column        | Description                                  |
| ------------- | -------------------------------------------- |
| `title`       | Title of the document/article                |
| `content`     | Main text content                            |
| `publishedAt` | Publication date (ISO 8601 format preferred) |

---

## 🚀 Quickstart

1. Prepare your data in `data/` (Parquet files with required columns).
2. Adjust parameters in [`configs/config.yaml`](configs/config.yaml) if needed.
3. Run:

   ```bash
   python -m src.llm_pipeline.main
   ```
4. Results are saved automatically to the Hydra run directory (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`).

> See [run.sh](run.sh) for an example SLURM job script, designed for the [ETH Zurich Euler cluster](https://ele.ethz.ch/resources-and-infrastructure/infrastructure/computational-ressources.html).

---

## 📜 Outputs

Each run produces:

* `final_outputs.csv` — classification results with reasoning traces
* Hydra config snapshot (`.hydra/`) for reproducibility
* Optional logs in the run directory

---

## 📝 Notes

* You can swap model providers by changing the `model` config group.
* To run multiple experiments, use Hydra’s `multirun` mode:

  ```bash
  python -m src.llm_pipeline.main -m model.temperature=0.3,0.5
  ```
