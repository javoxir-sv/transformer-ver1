# English-to-Italian Transformer

A small, from-scratch PyTorch implementation of an encoder-decoder Transformer for neural machine translation. The default configuration trains on the English–Italian split of the Hugging Face `Helsinki-NLP/opus_books` dataset.

The repository contains the model, data preparation, and training loop; it does not currently provide a separate command-line interface or a standalone translation/inference script.

## Contents

- [Features](#features)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Training and outputs](#training-and-outputs)
- [Model overview](#model-overview)
- [Current limitations](#current-limitations)

## Features

- Transformer encoder-decoder implemented with PyTorch modules in `model.py`.
- Word-level tokenizers trained with Hugging Face Tokenizers.
- Dataset download and loading through Hugging Face Datasets.
- Padding, start/end tokens, and attention masks for teacher-forced translation training.
- Greedy decoding of a small number of validation examples after each epoch.
- Per-epoch model and optimizer checkpoint files, plus TensorBoard training-loss logging.

## Project structure

| File | Purpose |
| --- | --- |
| `config.py` | Default hyperparameters, file paths, language pair, and checkpoint path helper. |
| `dataset.py` | `BilingualDataset`, which tokenizes sentence pairs, pads them to a fixed length, and creates masks. |
| `model.py` | Transformer embeddings, positional encoding, attention, encoder/decoder blocks, and model builder. |
| `train.py` | Dataset/tokenizer setup, training loop, validation sample generation, and checkpoint saving. |
| `requirements.txt` | Python package dependencies. |

## Requirements

- Python with a compatible PyTorch build. The dependency file does not pin a Python or package version.
- A supported PyTorch device: CUDA is used when available; otherwise training runs on CPU.
- Internet access on the first run to download the dataset and Python packages.
- Enough compute and memory for the selected batch size and sequence length. The defaults are aimed at GPU training; CPU training may be slow.

Install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorboard
```

`tensorboard` is used by PyTorch's `SummaryWriter`, but is not listed directly in `requirements.txt`.

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If you need a specific CUDA-enabled PyTorch build, install the appropriate PyTorch package for your machine before installing the remaining requirements.

## Getting started

From the repository root, start training with:

```bash
python train.py
```

The script selects CUDA when available, loads the configured dataset, creates or loads tokenizers, and begins training. On first use it downloads the dataset using Hugging Face Datasets and saves tokenizer JSON files in the current working directory.

Training writes its loss log through PyTorch's `SummaryWriter`. To monitor the log in a second terminal, run:

```bash
tensorboard --logdir runs
```

## Configuration

Defaults are returned by `get_config()` in `config.py`. Edit that function to change the run configuration:

| Setting | Default | Description |
| --- | --- | --- |
| `batch_size` | `26` | Training batch size. Memory use depends on the model, sequence length, and device. |
| `num_epochs` | `10` | Number of epochs to train. |
| `lr` | `1e-4` | Adam learning rate. |
| `seq_len` | `350` | Fixed source and target sequence length, including special tokens and padding. |
| `d_model` | `512` | Embedding and hidden-state width. |
| `src_lang` / `tgt_lang` | `"en"` / `"it"` | Language keys used to select the dataset configuration and translation text. |
| `model_dir` | `"weigths"` | Directory for model checkpoints. The default spelling is `weigths`. |
| `model_prefix` | `"tmodel_"` | Prefix for checkpoint filenames. |
| `preload` | `None` | Epoch label used to select a checkpoint file, e.g. `"07"`. See [Current limitations](#current-limitations). |
| `tokenizer_file` | `"tokenizer_{0}.json"` | Tokenizer filename template, formatted with the language code. |
| `experiment_name` | `"runs/tmodel"` | TensorBoard event directory. |
| `colab` | `False` | If enabled, the training loop runs `rsync` to copy `/content/transformer-ver1` to `/content/drive/MyDrive/transformer-ver1/` after each epoch. |

`seq_len` must be long enough for every sentence in both languages after tokenization. The current data loader raises an error if an example is too long; it does not truncate or filter long examples.

The tokenizer cache is keyed only by language code, not by dataset, tokenizer settings, or vocabulary version. If you change the dataset or tokenizer configuration, remove the relevant `tokenizer_<language>.json` file to rebuild that tokenizer.

## Training and outputs

`train.py` loads the `train` split for the configured `Helsinki-NLP/opus_books` language pair, builds or loads one tokenizer for each language, and creates a random 90/10 training/validation split. The split is not seeded, so it can differ between runs. Training uses shuffled batches; after each epoch, the script prints source, reference, and greedy-decoded translations for up to two validation examples. This is qualitative output, not a calculated BLEU or other translation metric.

By default, generated files are written relative to the directory from which `train.py` is run:

- `tokenizer_en.json` and `tokenizer_it.json` — word-level tokenizers, if not already present.
- `weigths/tmodel_00.pt`, `weigths/tmodel_01.pt`, … — per-epoch PyTorch checkpoint files.
- `runs/tmodel/` — TensorBoard event files.

Each checkpoint stores the epoch number, model state, optimizer state, and global training step. Checkpoints are not included in the repository.

## Model overview

The network is a standard encoder-decoder Transformer made from scratch in `model.py`. The default builder uses six encoder blocks and six decoder blocks, eight attention heads, a feed-forward width of 2048, and dropout of 0.1. The source and target embeddings are separate, positional encodings are sinusoidal, and a linear projection produces target-vocabulary log probabilities.

`dataset.py` prepares each example as follows:

1. Tokenize source and target text using separate word-level vocabularies.
2. Add `[SOS]` and `[EOS]` to the source; the decoder input starts with `[SOS]`, while the label ends with `[EOS]`.
3. Pad sequences to `seq_len` using `[PAD]`.
4. Create a source padding mask and a decoder mask that combines padding and causal masking.

## Current limitations

- **Checkpoint resume is incomplete:** setting `preload` loads the checkpoint's optimizer state and training counters, but `train.py` does not load the checkpoint's `model_state_dict`. Training therefore starts with newly initialized model weights rather than continuing from the saved model.
- **No standalone translation command:** greedy decoding is currently used only for the small validation sample printed during training.
- **Long examples stop the run:** the dataset rejects examples longer than `seq_len` instead of truncating or filtering them.
- **Run paths depend on the working directory:** tokenizers, checkpoints, and TensorBoard logs are written relative to the current working directory.
- **Colab copy behavior requires setup:** the optional copy uses `rsync` and assumes the source and mounted Google Drive paths shown in `config.py` exist.
