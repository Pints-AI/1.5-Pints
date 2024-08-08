# 1.5-Pints

This repo contains the model architecture, training scripts, and utilities of 1.5-Pints and 0.12-Pint, developed by Pints.AI.
By providing access to the model's codebase and architecture, this initiative seeks to facilitate the replication, experimentation, and further open-source development of Pint.

Join us at Discord: https://discord.gg/eGTRzDdH

 # Paper & Citation

 ```latex
 @misc{tan202415pintstechnicalreportpretraining,
       title={1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data}, 
       author={Calvin Tan and Jerome Wang},
       year={2024},
       eprint={2408.03506},
       archivePrefix={arXiv},
       primaryClass={cs.CL},
       url={https://arxiv.org/abs/2408.03506}, 
 }
 ```

# Installation

## Recommended OS

Typically just stick to `Ubuntu 22.04 LTS x86-64`. `Debian 12` has been tested to work as well.

_GOTCHA1_: Dont use `arm64` / `aarch64`. `xformers` does not support ARM64 processors.

_GOTCHA2_: We should not install system-wide CUDA using `apt`. It is best to constrain the CUDA installation to within the conda environment, so that different projects can use different CUDA versions.

## Install conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
sh Miniconda3-latest-Linux-x86_64.sh
```

Source just to be sure `conda` cli will be available:

```bash
source ~/.bashrc
```

Sometimes if you still face `conda: command cannot be found`, you can find the installation and source it:

`Note: This path assumes you took up the default installation settings. Otherwise, find where you installed it.`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

## Clone this repo

```bash
git clone https://github.com/Pints-App/Pints-Train.git && \
cd Pints-Train
```

## Create conda env

```bash
conda create --prefix ./.conda python=3.10 && \
conda activate ./.conda
```

`Note`: Stick to Python 3.10. 3.12 breaks a lot of things as of now (23 Feb 2024), and 3.11 has not been tested.

## Install CUDA toolkit

```bash
conda install nvidia/label/cuda-12.1.1::cuda-toolkit
```

## Install requirements

```bash
pip install -r requirements.txt && \
pip install flash-attn --no-build-isolation && \
pip install -r pretrain/requirements.txt
```

`Note`: The pip install for `dropout_layer_norm` can take up ~30 minutes to build depending on the machine.

### Install git-lfs if you haven't

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

### Download the pints-ai Expository Prose pretrain dataset

```bash
cd /path/to/dataset_dir
git clone https://huggingface.co/datasets/pints-ai/Expository-Prose-V1
```

### Prepare the dataset

```bash
python -m prepare_dataset.standard_parquet \
--source_path /path/to/dataset_dir  \
--train_val_split_ratio 0.9 \
--max_cores 60 \
--destination_path /path/to/output_dir
```

Refer to [prepare_dataset](https://github.com/Pints-App/Pints-Train/tree/master/prepare_dataset) folder for the dataset preparation scripts.

`max_cores` is not required if you don't OOM on high core machines.

# Train the model

## Lightning

```bash
fabric run model \
--accelerator=cuda \
--devices=8 \
pretrain/main.py \
--data_dir data/output \
--out_dir ../1.5-pints \
--gpus 8 \
--global_batch_size 512 \
--learning_rate 4e-4 \
--micro_batch_size 8 \
--max_step 96180 \
--warmup_steps 2000 \
--weight_decay 0.1 \
--beta1 0.9 \
--beta2 0.95 \
--grad_clip 1.0  \
--min_lr 4e-5 \
--model_name 1.5-Pints-2k \
--wandb_name <run_name> \
--wandb_project <project_name> \
--tokenizer_dir tokenizer/pints
```

`Note1`: `--devices` and `--gpus` must be the same. See `pretrain.py`'s `setup` arguments for all parameters that you can adjust.

`Note2`: Select the architecture (layers/dimensions/heads) configuration using `--model_name`. This must be in `lit_gpt/config.py`.

`Note3`: Select a `micro_batch_size` to optimize GPU memory. So far once started, it remains stable, even during validation. `micro_batch_size` need not be a number that `batch_size` is divisible by. `batch_size` is derived from `global_batch_size` / `devices`.

`Note4`: Modify `TRAIN_DATA_CONFIG` in `pretrain/main.py` to decide on the datasets used for training. Ensure that the dataset is [prepared](#prepare-the-dataset) beforehand.

## Wandb

If you are asked for the wandb API key, you can login and get from: <https://wandb.ai/authorize>

# Finetune the model

```bash
cd finetune && \
pip install -r requirements.txt
```

## Convert pretrained weights

Before you start finetuning, you need to convert the pretrain weights:

```bash
python convert/convert_pretrained_checkpoint.py --checkpoint_dir path/to/checkpoint --output_dir path/to/output
```

```bash
lightning run model \
--accelerator=cuda \
--devices=8 \
finetune/full.py \
--checkpoint_dir <path to lit_model.pth> \
--out_dir ~/1.5-pints-2k/ep2/step-00045000/finetuned \
--model_name 1.5-Pints-2k \
--gpus 8 \
--train.save_interval 6000 \
--train.global_batch_size 512 \
--train.micro_batch_size 8 \
--train.lr_warmup_steps 1125 \
--train.epoch 5 \
--train.learning_rate 2e-5 \
--train.max_seq_length 2048 \
--train.beta1 0.9 \
--train.beta2 0.95 \
--train.weight_decay 0.1 \
--logger_name wandb \
--tokenizer_dir tokenizer/pints \
--known_data_max_seq_length 2048 \
--wandb_project <project name>
```

# Run Direct Preference Optimization (DPO)

DPO is opted for use post-finetuning. See [here](https://github.com/Pints-App/Pints-Train/tree/master/dpo) for the execution process.

# Evaluate the model

See [here](https://github.com/Pints-App/Pints-Train/tree/master/eval)

# Use the model

## Convert lit models to HuggingFace models (pytorch and safetensors)

```bash
python convert_lit_to_hf.py \
--checkpoint_name lit_model.pth \
--directory ../models/1.5-pints \
--model_name 1.5-Pints-2k \
--output_config=True \
--safetensors=True \
--delete_pytorch_model=True
```

`Note`: We found better success using the `safetensors` file. Therefore it's recommended to use it instead of `pytorch_model.bin`.

## Use it with huggingface

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/model/path")
model = AutoModelForCausalLM.from_pretrained("/model/path")

prompt = '''<|im_start|>user
Do not go gentle into that good night.<|im_end|>
<|im_start|>assistant
'''

tokenized_input = tokenizer.encode(prompt)
tokenized_output = model.generate(tokenized_input)
print(tokenizer.decode(tokenized_output))
```

# Code Testing

This codebase comes with tests. If you need to make modifications, you can run the tests to ensure your modifications did not disrupt the existing code.

Install test requirements:

```bash
pip install -r requirements.test.txt
```

Run pytest:

```bash
python -m pytest --verbose
```
