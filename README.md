# Efficient Enhancement of  Small Language Models for Text-to-SQL via Fine-Tuning

MSc ML & AI project. Investigates how far a small language model's
text-to-SQL performance can be improved through fine-tuning, using the
Spider benchmark.

## Motivation and problem statement
LLMs are expensive and heavy to host due to paramater size.
So we turn to SLMs, however they are not as capable.
Then the question is how much can fine-tuning improve a SLM
(Qwen2.5-Coder-3B) on the Spider text-to-SQL task?


## Layout

### `common/`

Contains the shared infrastructure used across the experiments.

- `config.py` — loads the central experiment configuration.
- `data_loader.py` — loads Spider questions, reference SQL and database schemas.
- `schema_serializer.py` — converts database schemas into text for the model prompt.
- `prompt_builder.py` — constructs the schema + question instruction presented to the model.
- `model.py` — loads Qwen models and performs SQL generation.
- `runner.py` — runs the complete evaluation loop for each Spider example.
- `db_executor.py` — executes generated and reference SQL against the Spider SQLite databases.
- `evaluation.py` — compares execution results and calculates execution accuracy.
- `profiler.py` — records generation timing information.

### `training_scripts/`

Contains the PEFT training and evaluation scripts.

- `train_lora.py` — trains the LoRA adapter on the Spider training split.
- `run_slm_lora.py` — evaluates the trained LoRA model.
- `train_ia3.py` — trains the IA³ adapter on the Spider training split.
- `run_slm_ia3.py` — evaluates the trained IA³ model.

### `baselines/`

Contains the unchanged 3B baseline evaluation used as the reference point for
the PEFT experiments.

### `notebooks/`

Contains the Colab notebooks used to run the experiments.

- `Vanilla_SLM_Experiment.ipynb` — 3B vanilla baseline.
- `LoRA_Experiment.ipynb` — LoRA training and evaluation.
- `IA3_Experiment.ipynb` — IA³ training and evaluation.
- `14B_Experiment.ipynb` — 14B variant for comparison.

The notebooks are where the experiments are actually ran, while the underlying
training, generation and evaluation logic remains in the Python modules above.

## Dataset Setup

The Spider dataset is not included in this repository.

Download the Spider dataset from its official distribution and place the files
in the following structure:

data/spider/
├── dev.json
├── train_spider.json
├── tables.json
└── database/


