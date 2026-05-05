# Visual-noise Guided In-context Distillation for Multimodal Large Language Model Unlearning (VGID)

This repository is a working benchmark and training codebase for multimodal unlearning experiments built around the MLLMU-Bench data format. It keeps the standard fine-tuning and evaluation scripts from the benchmark workflow, and adds a dedicated `VGID` baseline implementation in [baselines/Vgid.py](/Users/junkaichen/Downloads/code/VGID/baselines/Vgid.py). At the moment, the method should be considered supported only for LLaVA.

The repository is organized around three tasks:

- Train a vanilla multimodal model on the benchmark training data.
- Train unlearning baselines, especially `VGID`.
- Evaluate forget / retain / test performance with the provided evaluation scripts.

## Repository Layout

- [finetune.py](/Users/junkaichen/Downloads/code/VGID/finetune.py): fine-tune a vanilla model from parquet training data.
- [baselines/Vgid.py](/Users/junkaichen/Downloads/code/VGID/baselines/Vgid.py): the main VGID training script.
- [baselines/README.md](/Users/junkaichen/Downloads/code/VGID/baselines/README.md): usage examples for the other baselines.
- [eval.py](/Users/junkaichen/Downloads/code/VGID/eval.py): benchmark evaluation script.
- [eval_gpt.py](/Users/junkaichen/Downloads/code/VGID/eval_gpt.py): factuality scoring with the OpenAI API.
- [data_process/data_preprocess.py](/Users/junkaichen/Downloads/code/VGID/data_process/data_preprocess.py): standard parquet dataset processing.
- [data_process/data_preprocess_ic.py](/Users/junkaichen/Downloads/code/VGID/data_process/data_preprocess_ic.py): dataset processing used by `VGID`.

## Environment

Recommended environment:

```bash
conda create -n vgid python=3.10
conda activate vgid
pip install -r requirements.txt
```

The current dependency set is centered on:

- `torch==2.4.0`
- `transformers==4.45.1`
- `accelerate==0.33.0`
- `peft==0.12.0`
- `datasets==2.21.0`

## Data Preparation

This codebase expects the same parquet-based data layout used by MLLMU-Bench. In practice, you need two kinds of data:

1. Full training data for vanilla fine-tuning.
2. Forget / retain split data for unlearning baselines.

Typical split directory layout:

```text
DATA_SPLIT_DIR/
  forget_5/
    train-00000-of-00001.parquet
  retain_95/
    train-00000-of-00001.parquet
```

The scripts load image bytes and metadata directly from parquet files, so no separate image-folder preprocessing is required by default.

## Vanilla Fine-Tuning

You can first train a vanilla model with [finetune.py](/Users/junkaichen/Downloads/code/VGID/finetune.py):

```bash
python finetune.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --save_dir /path/to/vanilla_model \
  --data_dir /path/to/train-00000-of-00001.parquet \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 5 \
  --max_length 384
```

Currently supported model family for this method:

- `llava-*`

## VGID Baseline

The main script is [baselines/Vgid.py](/Users/junkaichen/Downloads/code/VGID/baselines/Vgid.py).

Current support status:

- `VGID` should be used with LLaVA checkpoints.
- Although some code paths still contain references to other model families, this repository should currently be treated as LLaVA-only for actual use and reproduction.

### What the Script Does

`Vgid.py` trains on two streams at the same time:

- A `forget` stream, intended to suppress memorized private information.
- A `retain` stream, intended to preserve the original model behavior on retained data.

The script:

- Loads a vanilla model from `--vanilla_dir`.
- Clones it as a frozen teacher/reference (`model_vanilla`).
- Adds LoRA adapters to the trainable student model.
- Alternates one forget batch and one retain batch during training.
- Saves checkpoints after every epoch and also writes the final model to `--save_dir`.

### Loss Structure

The implementation uses two losses:

- `loss_forget`: KL divergence between the student prediction on the original forget input and the vanilla teacher prediction on a noise-corrupted image.
- `loss_retain`: retain-side preservation loss.

This repository version also supports explicit weighting of the two branches:

- `--alpha`: weight for `loss_forget`
- `--beta`: weight for `loss_retain`

The backward pass is effectively scaled as:

```text
weighted_forget_loss = alpha * loss_forget
weighted_retain_loss = beta * loss_retain
```

This makes it easy to control the forget / retain tradeoff during training.

### Basic Usage

```bash
python baselines/Vgid.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --vanilla_dir /path/to/vanilla_model \
  --data_split_dir /path/to/split_data \
  --forget_split_ratio 5 \
  --save_dir /path/to/vgid_output \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 1 \
  --max_length 384
```

### Usage with Loss Weights

Example with stronger forgetting pressure and weaker retain preservation:

```bash
python baselines/Vgid.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --vanilla_dir /path/to/vanilla_model \
  --data_split_dir /path/to/split_data \
  --forget_split_ratio 5 \
  --save_dir /path/to/vgid_output \
  --batch_size 4 \
  --lr 2e-5 \
  --num_epochs 1 \
  --max_length 384 \
  --alpha 2.0 \
  --beta 0.5
```

### Arguments

- `--model_id`: model family identifier. Example: `llava-hf/llava-1.5-7b-hf`.
- `--vanilla_dir`: local path to the vanilla model checkpoint used to initialize the student and teacher.
- `--save_dir`: output directory for epoch checkpoints and the final model.
- `--data_split_dir`: root folder that contains `forget_x` and `retain_y` parquet folders.
- `--forget_split_ratio`: forget percentage used to resolve the split folders. Example: `5` maps to `forget_5` and `retain_95`.
- `--batch_size`: training batch size.
- `--lr`: learning rate.
- `--alpha`: coefficient for `loss_forget`. Default: `1.0`.
- `--beta`: coefficient for `loss_retain`. Default: `1.0`.
- `--num_epochs`: number of training epochs.
- `--max_length`: text sequence truncation length used by the collate function.
- `--gradient_accumulation`: enable gradient accumulation logic in the script.
- `--trainer`: unused compatibility flag kept by the original code.

### Outputs

`Vgid.py` writes:

- Per-epoch checkpoints in `SAVE_DIR/epoch_1`, `SAVE_DIR/epoch_2`, ...
- Final model weights in `SAVE_DIR`

### Practical Notes

- `--vanilla_dir` must point to a local model directory. The script loads it with `local_files_only=True` for the LLaVA path.
- The script expects parquet files named `train-00000-of-00001.parquet` in the forget and retain folders.
- The implementation uses LoRA for training and saves the resulting adapted model weights with `save_pretrained`.
- If you are tuning `alpha` and `beta`, start from `1.0 / 1.0`, then increase `alpha` to encourage more aggressive forgetting.

## Other Baselines

The repository also includes:

- [baselines/GA.py](/Users/junkaichen/Downloads/code/VGID/baselines/GA.py)
- [baselines/GA_Difference.py](/Users/junkaichen/Downloads/code/VGID/baselines/GA_Difference.py)
- [baselines/KL_Min.py](/Users/junkaichen/Downloads/code/VGID/baselines/KL_Min.py)
- [baselines/NPO.py](/Users/junkaichen/Downloads/code/VGID/baselines/NPO.py)
- [baselines/reference_model_FT.py](/Users/junkaichen/Downloads/code/VGID/baselines/reference_model_FT.py)

Command examples for those scripts are in [baselines/README.md](/Users/junkaichen/Downloads/code/VGID/baselines/README.md).

## Evaluation

After training a vanilla or unlearned model, evaluate it with [eval.py](/Users/junkaichen/Downloads/code/VGID/eval.py):

```bash
python eval.py \
  --model_id llava-hf/llava-1.5-7b-hf \
  --cache_path /path/to/model_checkpoint \
  --data_split_folder /path/to/split_data \
  --few_shot_data /path/to/few_shot.parquet \
  --test_data /path/to/test_set \
  --celebrity_data /path/to/celebrity.parquet \
  --output_folder /path/to/eval_output \
  --output_file run_name \
  --forget_ratio 5
```

For factuality scoring of generated answers, use [eval_gpt.py](/Users/junkaichen/Downloads/code/VGID/eval_gpt.py) after setting `OPENAI_API_KEY`.

## Workflow Summary

1. Prepare parquet training data and forget/retain split data.
2. Train a vanilla model with `finetune.py`.
3. Run `baselines/Vgid.py` with your chosen `alpha` and `beta`.
4. Evaluate the saved checkpoint with `eval.py`.
5. Optionally score generations with `eval_gpt.py`.
