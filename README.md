# Post-Training & Fine-Tuning LLMs

This repository includes Python Jupyter notebooks that implement and demonstrate post-training and fine-tuning methods for large language models (LLMs). The notebooks explore both parameter-efficient fine-tuning (PEFT) strategies (e.g., LoRA) and reinforcement-based fine-tuning approaches (e.g., PPO / GRPO for RLHF-style optimization). The goal is to provide reproducible examples and practical guidance for adapting pre-trained LLMs to downstream tasks and reward-driven behavior.

- Repository: https://github.com/cetinkayaevren/post_train_fine_tune_llms
- Primary language / artifacts: Python - Jupyter Notebooks

Table of contents
- About
- Repository structure
- What’s included (notebook overviews)
- Requirements
- Quick start
- Typical workflows
  - PEFT (LoRA) + supervised fine-tuning
  - RL-based fine-tuning (PPO / GRPO)
- Data & checkpoints
- Evaluation & logging
- Reproducibility tips
- Contributing
- References
- License & contact

## About
This collection demonstrates practical recipes and experiments for adapting LLMs:
- Post-training and lightweight fine-tuning with PEFT (LoRA) and Hugging Face ecosystem tools.
- Reinforcement-style fine-tuning using PPO and GRPO variants to optimize reward signals (RLHF-style).
- Notebook-driven experiments that can be adapted for different base models, datasets, and compute budgets.

## Repository structure
- GRPO_Reinforcement_Fine_Tuning_LLMs/
  - Notebooks and helper code that implement GRPO-based reinforcement fine-tuning workflows for LLMs.
- RLHF_PPO_PEFT_LoRA_Fine_Tuning_LLMs/
  - Notebooks and examples showing RLHF pipelines with PPO plus PEFT/LoRA and related utilities.
- README.md
  - This file (overview, usage, and instructions).

## What’s included (high level)
- End-to-end notebook examples that:
  - Load and prepare datasets (Hugging Face datasets or local data).
  - Initialize pretrained models and tokenizers.
  - Apply PEFT techniques (LoRA) for parameter-efficient fine-tuning.
  - Implement RL-based fine-tuning loops (PPO, GRPO) to optimize reward models.
  - Track training and evaluation metrics; optionally save model checkpoints.
- Guidance on environment setup and package dependencies.

## Requirements
(These are typical packages used in LLM fine-tuning workflows — adapt as needed for the notebooks.)
- Python 3.8+
- PyTorch (recommended: 1.12+ or a version compatible with CUDA drivers)
- transformers
- accelerate
- datasets
- peft
- bitsandbytes (for 8-bit training / memory efficient adapters)
- trl (or trlX) for RLHF/PPO utilities
- jupyterlab or notebook
- numpy, pandas, scikit-learn
- matplotlib / seaborn (optional for plotting)
- optuna (optional — for hyperparameter tuning)
Install example (pip)
```
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
.\.venv\Scripts\activate          # Windows (PowerShell)
pip install --upgrade pip
pip install jupyterlab torch transformers accelerate datasets peft bitsandbytes trl numpy pandas scikit-learn matplotlib seaborn
```
If you prefer conda, create an environment and install using conda-forge/pip as appropriate.

## Quick start
1. Clone the repository:
   git clone https://github.com/cetinkayaevren/post_train_fine_tune_llms.git
2. Create and activate your Python environment (see Requirements).
3. Start Jupyter:
   jupyter lab
4. Open the notebook(s) under:
   - GRPO_Reinforcement_Fine_Tuning_LLMs/
   - RLHF_PPO_PEFT_LoRA_Fine_Tuning_LLMs/
5. Read the notebook top cells for per-notebook requirements and the expected dataset/paths.

## Typical workflows

PEFT (LoRA) + supervised fine-tuning
- Purpose: Quickly adapt a large pre-trained model to a downstream task using a small number of trainable parameters (LoRA).
- Typical steps in notebook:
  - Load pre-trained model + tokenizer (HF transformers).
  - Apply PEFT/LoRA wrappers to the model.
  - Prepare dataset (tokenize, collate).
  - Train with Hugging Face Trainer or custom loop using accelerate.
  - Save final LoRA adapter + tokenizer and optionally merge adapters into base model.

## Reinforcement-based fine-tuning (PPO / GRPO / RLHF)
- Purpose: Optimize model outputs according to a reward function (e.g., preference model, heuristics).
- Typical steps in notebook:
  - Prepare a reward model or reward function.
  - Use a baseline policy (pretrained model); optionally apply PEFT to reduce trainable parameters.
  - Run PPO or GRPO loop to update model parameters to maximize expected reward while keeping divergence from base policy limited.
  - Log reward statistics and sample generations to inspect behavior changes.

## Data & checkpoints
- Notebooks are written to work with Hugging Face datasets and local data (CSV/JSON). Check the top of each notebook for dataset loading instructions.
- Use a checkpointing directory and save:
  - model weights / adapter weights
  - tokenizer files
  - training logs / metrics
- For large models, consider saving only adapters (LoRA) to minimize storage needs.

## Evaluation & logging
- Recommended metrics:
  - Task-specific metrics (accuracy, F1, BLEU, ROUGE, etc.)
  - Perplexity (for language modeling)
  - Reward, reward variance, and KL/divergence for RL-based methods
- Recommended logging tools:
  - TensorBoard
  - Weights & Biases (W&B)
  - Simple CSV/JSON logs for reproducibility

## Reproducibility tips
- Pin package versions in a requirements.txt or environment.yml.
- Set random seeds (PyTorch, NumPy, Python random) at the top of notebooks.
- Log hardware, CUDA/cuDNN versions, and model configurations.
- Save the exact model and tokenizer checkpoints used for evaluation.

## Performance & compute notes
- GPU (preferably with >12GB VRAM) is strongly recommended for all model fine-tuning experiments.
- For memory-limited setups:
  - Use 8-bit optimizers (bitsandbytes).
  - Use PEFT (LoRA) to reduce trainable parameters.
  - Lower batch sizes and use gradient accumulation.
- When using bitsandbytes and 8-bit training, ensure compatibility between CUDA, PyTorch, and the bitsandbytes release.

## Contributing
- Contributions and improvements are welcome. Suggested ways to contribute:
  - Add missing instructions for notebook-specific dependencies.
  - Provide small reproducible datasets or colab / binder examples.
  - Improve logging, add model-card templates, or include evaluation notebooks.
- Open issues and PRs in the repository with clear descriptions and test instructions.

## References
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PEFT (LoRA): https://github.com/huggingface/peft
- RLHF / PPO utilities: trl / trlX repositories
- BitsandBytes for memory-efficient training: https://github.com/facebookresearch/bitsandbytes

## License & contact
- Add an appropriate license file if you plan to make this repository reusable (MIT, Apache-2.0, etc.).
- For questions or collaboration: @cetinkayaevren (GitHub) — https://github.com/cetinkayaevren

### Notes
- The notebooks themselves include per-file details and execution instructions. Open each notebook to see dataset paths, configurable hyperparameters, and experimental notes.
- If you want, I can:
  - generate a requirements.txt or environment.yml for the notebooks,
  - produce per-notebook README summaries (one-paragraph each) after you point me to specific notebook filenames,
  - or create a short badge/usage section (e.g., Colab links) to make sharing easier.
