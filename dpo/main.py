# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Note: Script was adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py
# Support running without installing as a package
# ruff: noqa: E402
import sys
from pathlib import Path

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    IntervalStrategy,
    TrainingArguments,
)
from trl import (
    DPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dpo.adapters.base import DatasetAdapter
from dpo.adapters.distilabel_capybara_dpo import DistilabelCapybaraDPO
from dpo.adapters.nectar import Nectar
from dpo.adapters.ultrafeedback_binarized import UltraFeedbackBinarized
from dpo.make_dataset import make_dataset

DEEPSPEED_LOCAL_RANK: Union[None, int] = None

# StableLM and Zephyr recipe: https://arxiv.org/pdf/2310.16944.pdf
# Note: Global batch size of 32 needs to be controlled by --per_device_train_batch_size and --gradient_accumulation_steps
#       depending on the number of GPUs available: per_device_train_batch_size * gradient_accumulation_steps = 32 / GPUs
BETA = 0.1
# LR_SCHEDULER_TYPE = 'linear' # TrainingArgs already default to Linear
WARMUP_RATIO = 0.1
LEARNING_RATE = 5e-7


try:
    from deepspeed.comm import get_local_rank

    DEEPSPEED_LOCAL_RANK = get_local_rank()

except AssertionError as error:
    # This error is expected if we are not running DeepSpeed
    expected_error = (
        'DeepSpeed backend not set, please initialize it using init_process_group()'
    )
    if str(error) != expected_error:
        raise error

except ImportError:
    if __name__ == '__main__':
        print('WARN: DeepSpeed is not installed. Continuing script for debugging.')


@dataclass
class ScriptArguments:
    beta: float = field(
        default=BETA, metadata={'help': 'the beta parameter for DPO loss'}
    )
    max_length: int = field(
        default=2048, metadata={'help': 'max length of each sample'}
    )
    max_prompt_length: int = field(
        default=512, metadata={'help': "max length of each sample's prompt"}
    )
    max_target_length: int = field(
        default=1536,
        metadata={
            'help': "Only used for encoder decoder model. Max target of each sample's prompt"
        },
    )
    sanity_check: bool = field(
        default=True, metadata={'help': 'only train on 1000 samples'}
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            'help': 'debug argument for distributed training;'
            'fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See'
            'https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992'
        },
    )
    generate_during_eval: bool = field(
        default=False, metadata={'help': 'Generate during evaluation'}
    )


SEED = 8888
DTYPE = 'bfloat16'


@dataclass
class PintsTrainingArguments(TrainingArguments):
    seed: int = field(
        default=SEED,
        metadata={'help': 'Random seed that will be set at the beginning of training.'},
    )
    data_seed: int = field(
        default=SEED, metadata={'help': 'Random seed to be used with data samplers.'}
    )
    learning_rate: float = field(
        default=LEARNING_RATE, metadata={'help': 'The initial learning rate for AdamW.'}
    )

    warmup_ratio: float = field(
        default=WARMUP_RATIO,
        metadata={'help': 'Linear warmup over warmup_ratio fraction of total steps.'},
    )
    num_train_epochs: float = field(
        default=5, metadata={'help': 'Total number of training epochs to perform.'}
    )
    save_strategy: Union[IntervalStrategy, str] = field(
        default='epoch',
        metadata={'help': 'The checkpoint save strategy to use.'},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            'help': (
                'Log every X updates steps. Should be an integer or a float in range `[0,1)`. '
                'If smaller than 1, will be interpreted as ratio of total training steps.'
            )
        },
    )
    bf16: bool = field(
        default=True if DTYPE == 'bfloat16' else False,
        metadata={
            'help': (
                'Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA'
                ' architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change.'
            )
        },
    )

    # TODO: Is it possible to toggle this on/off using CLI? Tried and failed, as HfArgumentParser
    #       will parse `--deepspeed None` or `False` into 'None' and 'False' string types,
    #       which are taken is legit file paths, triggering json parsing failures.
    #
    # DEEPSPEED - use if model cannot fit into GPUs. It slows down training.
    # To use DeepSpeed, you can either use `--deepspeed ds_config` or uncomment this.
    # deepspeed: Optional[str] = field(
    #     default=f'{str(Path(__file__).parent / "ds_config_stage1.json")}',
    #     metadata={
    #         'help': (
    #             'Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already'
    #             ' loaded json file as a dict'
    #         )
    #     },
    # )

    # We are using `DPODataCollatorWithPadding` as data collator by default. It needs `remove_unused_columns=False`.
    # This is per the warning if `remove_unused_columns` is not defined and defaults to True:
    # When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments we have set it for you, but you should do it yourself in the future.
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'Remove columns not required by the model when using an nlp.Dataset.'
        },
    )


@dataclass
class PintsModelConfig(ModelConfig):
    attn_implementation: Optional[str] = field(
        default='flash_attention_2',
        metadata={
            'help': (
                'Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`'
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=DTYPE,
        metadata={
            'help': (
                'Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the '
                "dtype will be automatically derived from the model's weights."
            ),
            'choices': ['auto', 'bfloat16', 'float16', 'float32'],
        },
    )

# if __name__ == '__main__':
#     adapters: List[DatasetAdapter] = [
#         # DistilabelCapybaraDPO(),
#         # OrcaDPOPairs(),
#         Nectar(seed=SEED),
#         # OpenHermesPreferences(),
#         # UltraFeedbackBinarized(score_distance=0),
#     ]

#     train_dataset, eval_dataset = make_dataset(adapters, seed=SEED)
#     for row in train_dataset:
#         print('=' * 20 + 'PROMPT' + '=' * 20)
#         print(row['prompt'])
#         print('=' * 20 + 'CHOSEN' + '=' * 20)
#         print(row['chosen'])
#         print('=' * 20 + 'REJECTED' + '=' * 20)
#         print(row['rejected'])
#         exit()

# exit()

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ScriptArguments, PintsTrainingArguments, PintsModelConfig)
    )
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    # Print out the arguments and training settings/hyperparameters
    if (
        # if using deepspeed, we only print for the first rank
        isinstance(DEEPSPEED_LOCAL_RANK, int)
        and DEEPSPEED_LOCAL_RANK == 0
        or
        # it is not using deepspeed, just print
        DEEPSPEED_LOCAL_RANK is None
    ):
        print('======================')
        print(training_args)
        print('======================')
        print(model_config)
        print('======================')
        print(args)
        print('======================')

    ####################
    # Model & Tokenizer
    ####################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ['auto', None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else 'cuda',
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            **model_kwargs
        )
    else:
        model_ref = None

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ##################
    # Datasets to use
    ##################
    adapters: List[DatasetAdapter] = [
        # DistilabelCapybaraDPO(),
        # OrcaDPOPairs(),
        Nectar(seed=SEED),
        # OpenHermesPreferences(),
        # UltraFeedbackBinarized(score_distance=0),
    ]

    train_dataset, eval_dataset = make_dataset(adapters, seed=SEED)

    # DPO is highly sensitive to quality of data.
    # Naively truncating data is not ideal, so we want to tokenize and omit
    # those that exceed max_length
    # TODO: May DPOTrainer can take in tokenized data directly.
    #       So that we don't need to tokenize here and then later inside of DPOTrainer
    train_dataset_subset = []
    for row in train_dataset:
        prompt_length = len(tokenizer.encode(row['prompt']))
        chosen_length = len(tokenizer.encode(row['chosen']))
        rejected_length = len(tokenizer.encode(row['rejected']))

        if (
            prompt_length + chosen_length > args.max_length
            or prompt_length + rejected_length > args.max_length
        ):
            continue

        train_dataset_subset.append(row)

    eval_dataset_subset = []
    for row in eval_dataset:
        prompt_length = len(tokenizer.encode(row['prompt']))
        chosen_length = len(tokenizer.encode(row['chosen']))
        rejected_length = len(tokenizer.encode(row['rejected']))

        remaining_length = args.max_length - prompt_length

        if (
            remaining_length - chosen_length < 0
            or remaining_length - rejected_length < 0
        ):
            continue

        eval_dataset_subset.append(row)

    print('Final dataset size:')
    print(f'Train dataset [{len(train_dataset_subset)}]')
    print(f'Eval dataset [{len(eval_dataset_subset)}]')

    # The prompt length is already handled by `max_length` parameter
    # Setting this as no futher truncation is required
    args.max_target_length = 9999999
    args.max_prompt_length = 9999999

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=Dataset.from_dict(
            {
                'prompt': [row['prompt'] for row in train_dataset_subset],
                'chosen': [row['chosen'] for row in train_dataset_subset],
                'rejected': [row['rejected'] for row in train_dataset_subset],
            }
        ),
        eval_dataset=Dataset.from_dict(
            {
                'prompt': [row['prompt'] for row in eval_dataset_subset],
                'chosen': [row['chosen'] for row in eval_dataset_subset],
                'rejected': [row['rejected'] for row in eval_dataset_subset],
            }
        ),
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_target_length=args.max_target_length,
        max_prompt_length=args.max_prompt_length,
        generate_during_eval=args.generate_during_eval,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
