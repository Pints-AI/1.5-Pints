# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modifications:
# TODO: List modifications

# Support running without installing as a package
# ruff: noqa: E402
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import dataclasses
import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union

import lightning
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import RunningMean

from lit_gpt.args import EvalArgs, TrainArgs

from lit_gpt.datamodules.capybara import Capybara
from lit_gpt.datamodules.deita import Deita
from lit_gpt.datamodules.llama_instruct import LlamaInstruct
from lit_gpt.datamodules.meta_math_qa import MetaMathQA
from lit_gpt.datamodules.slim_orca_dedup import SlimOrcaDedup
from lit_gpt.datamodules.ultrachat_200k import UltraChat
from lit_gpt.datamodules.wizardlm_evol_instruct_v2 import WizardLMEvolInstructV2

from lit_gpt.datamodules.base import DataModule, get_sft_collate_fn
from lit_gpt.generate.base import generate
from lit_gpt.model import GPT, Block, Config
from lit_gpt.prompts import save_prompt_style
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
    parse_devices,
    save_hyperparameters,
    humanize_seconds,
)


class MovingAverage:
    moving_average = 0
    _samples = 0

    def compute(self, value: float):
        new_total = self.moving_average * self._samples + value
        self._samples += 1
        self.moving_average = new_total / self._samples
        return self.moving_average


MOVING_AVERAGE = MovingAverage()
TOTAL_ITERATIONS = None


def setup(
    checkpoint_dir: Path,
    out_dir: Path,
    model_name: str,
    precision: Optional[str] = None,
    gpus: Union[int, str] = 1,
    resume: Union[bool, Path] = False,
    data: Optional[List[DataModule]] = None,
    train: TrainArgs = TrainArgs(
        save_interval=6000,
        log_interval=1,
        # Hyperparameters follow Zephyr Alignment Handbook. See https://arxiv.org/pdf/2310.16944
        global_batch_size=512,
        micro_batch_size=8,
        lr_warmup_steps=1125,  # 10% of the steps of the full dataset
        epochs=5,
        learning_rate=2e-5,
        max_seq_length=2048,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
    ),
    eval: EvalArgs = EvalArgs(interval=600, max_new_tokens=100, max_iters=100),
    logger_name: Literal['wandb', 'tensorboard', 'csv'] = 'csv',
    seed: int = 8888,  # Maintain this for reproducibility training data sequence.
    tokenizer_dir: Optional[Path] = None,
    known_data_max_seq_length: Optional[int] = None,
    wandb_project: Optional[str] = None,
) -> None:
    """Finetune a model.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        tokenizer_dir: The directory where your tokenizers are. If not provided, will attempt to load tokenizers from `checkpoint_dir`.
        known_data_max_seq_length: Provide this if you already know what's the max sequence length in your dataset. This will skip over the search for max sequence in data, saving a lot of time.
        wandb_project: The name of the wandb project name.
    """

    # Some bug that doesn't recognise devices as a CLI command
    devices = gpus

    pprint(locals())
    if data is None:
        data = [
            Capybara(),
            Deita(),
            LlamaInstruct(),
            MetaMathQA(),
            SlimOrcaDedup(),
            UltraChat(),
            WizardLMEvolInstructV2(),
        ]

    devices = parse_devices(devices)

    # TODO: Reconcile the older pretrain code with this new finetune code to enable this.
    # check_valid_checkpoint_dir(checkpoint_dir)
    # config = Config.from_file(checkpoint_dir / 'model_config.yaml')
    config = Config.from_name(model_name)

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        project_name=wandb_project
        if wandb_project is not None
        else f'finetune-{config.name}',
        resume=resume,
        log_interval=train.log_interval,
    )

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            # xformers SwiGLU is not compatible with activation checkpointing.
            # See https://github.com/jzhang38/TinyLlama/issues/152
            activation_checkpointing_policy=None,
            # activation_checkpointing_policy={Block},
            state_dict_type='full',
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = 'auto'

    fabric = lightning.Fabric(
        devices=devices, strategy=strategy, precision=precision, loggers=logger
    )

    # If sequence length is not defined, we set based on block size.
    if train.max_seq_length is None:
        fabric.print(
            f'`max_seq_length` is not defined. Defaulting to `config.block_size` of [{config.block_size}].'
        )
        train.max_seq_length = config.block_size

    main(
        fabric,
        devices,
        resume,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        tokenizer_dir,
        known_data_max_seq_length,
    )

    # NOTE: We can conditionally launch if this code is not run from CLI.
    # fabric.launch(
    #     main,
    #     devices,
    #     resume,
    #     seed,
    #     config,
    #     data,
    #     checkpoint_dir,
    #     out_dir,
    #     train,
    #     eval,
    #     tokenizer_dir,
    # )


def main(
    fabric: lightning.Fabric,
    devices: int,
    resume: Union[bool, Path],
    seed: int,
    config: Config,
    data: List[DataModule],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    tokenizer_dir: Optional[str],
    known_data_max_seq_length: Optional[int],
) -> None:
    validate_args(train, eval)

    if tokenizer_dir is None:
        tokenizer_dir = checkpoint_dir

    tokenizer = Tokenizer(tokenizer_dir)
    train_dataloader, val_dataloader = get_dataloaders(
        fabric, data, tokenizer, train, seed
    )

    iterations_per_epoch = len(train_dataloader)
    global TOTAL_ITERATIONS
    TOTAL_ITERATIONS = iterations_per_epoch * train.epochs
    gradient_accumulation_iters = train.gradient_accumulation_iters(devices)
    steps_per_epoch = iterations_per_epoch // gradient_accumulation_iters

    fabric.print(f'Total iterations: [{TOTAL_ITERATIONS}]')
    fabric.print(f'Iterations per epoch: [{iterations_per_epoch}]')
    fabric.print(f'Iterations per step: [{gradient_accumulation_iters}]')
    fabric.print(f'Total steps: [{TOTAL_ITERATIONS/gradient_accumulation_iters}]')
    fabric.print(f'Steps per epoch: [{steps_per_epoch}]')

    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float('inf'))
    )

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = checkpoint_dir / 'lit_model.pth'
    with fabric.init_module(empty_init=(devices > 1)):
        model = GPT(config)

    fabric.print(
        f'Number of trainable parameters: {num_parameters(model, requires_grad=True):,}'
    )

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )
    state = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'iter_num': 0,
        'step_count': 0,
    }

    if resume is True:
        resume = max(
            out_dir.rglob('step-*/*.pth'),
            key=(lambda p: int(p.parent.name.split('-')[1])),
        )
    if resume:
        fabric.print(f'Resuming training from {resume}')
        fabric.load(resume, state)
    else:
        load_checkpoint(fabric, state['model'], checkpoint_path)

    train_time = time.perf_counter()
    fit(
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        devices,
        resume,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
        tokenizer_dir,
        known_data_max_seq_length,
    )
    fabric.print(f'Training time: {(time.perf_counter()-train_time):.2f}s')
    if fabric.device.type == 'cuda':
        fabric.print(f'Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB')

    # Save the final checkpoint at the end of training
    save_path = out_dir / 'final' / 'lit_model.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(save_path, {'model': state['model']})
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data[0].prompt_style, save_path.parent)


def fit(
    fabric: lightning.Fabric,
    state: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    resume: Union[bool, Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: List[DataModule],
    tokenizer_dir: Optional[Path],
    known_data_max_seq_length: Optional[int],
) -> None:
    model = state['model']
    optimizer = state['optimizer']
    scheduler = state['scheduler']

    if tokenizer_dir is None:
        tokenizer_dir = checkpoint_dir

    tokenizer = Tokenizer(tokenizer_dir)

    # If user provided a known sequence length of the dataset, save time by setting it
    # instead of iterating through the dataset to find it.
    if known_data_max_seq_length is not None:
        # If both `train.max_seq_length` and `known_data_max_seq_length` is provided
        # will need to check which is the lower one to use.
        if (
            train.max_seq_length is not None
            and known_data_max_seq_length > train.max_seq_length
        ):
            message = 'WARN: You specified a `known_data_max_seq_length` that is more than `train.max_seq_length`.'
            message = " The model's max sequence length will fallback to `train.max_seq_length`."
            fabric.print(message)

            model.max_seq_length = train.max_seq_length
            fabric.print(
                f'Model `max_seq_length` set to `train.max_seq_length` of [{train.max_seq_length}].'
            )
        else:
            model.max_seq_length = known_data_max_seq_length
            fabric.print(
                f'Model `max_seq_length` set to `known_data_max_seq_length` of [{known_data_max_seq_length}].'
            )

    else:
        # User does not know, so we have to find out:
        message = 'Getting longest sequence length from dataset to set the training max sequence length.'
        message += ' If sequences from dataset is short, using a short max sequence speeds up training time.'
        fabric.print(message)

        longest_seq_length, longest_seq_ix = get_longest_seq_length(
            train_dataloader.dataset
        )
        model.max_seq_length = min(
            longest_seq_length, train.max_seq_length or float('inf')
        )
        fabric.print(
            f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
            f' {model.max_seq_length} and context length is {model.config.block_size}'
        )

    validate(
        fabric,
        model,
        val_dataloader,
        tokenizer,
        dataclasses.replace(eval, max_iters=2),
        data,
    )  # sanity check
    initial_iter = state['iter_num']
    max_steps = train.max_steps or float('inf')
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
    # TODO: Save train_dataloader to state. Need to verify how much disk space
    #       that consumes.
    #       For implementation, see https://github.com/Lightning-AI/litgpt/blob/67614e9d71ecc8138986f17a672cff36004a3710/litgpt/pretrain.py#L207-L213
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f'Resuming dataset: {resume_iter} / {initial_iter}')
        fabric.barrier()
        fabric.print(
            f'Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration'
            f' {initial_iter}.'
        )

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    fabric.barrier()
    val_loss = 'n/a'

    while state['step_count'] < max_steps and train_iterator.epoch < train.epochs:
        state['iter_num'] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        input_ids, targets = batch['input_ids'], batch['labels']

        is_accumulating = (
            state['iter_num'] % train.gradient_accumulation_iters(devices) != 0
        )
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            # shift the targets such that output n predicts token n+1
            loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state['step_count'] += 1

        if state['iter_num'] % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            metrics = {
                'loss': loss,
                'iter': state['iter_num'],
                'step': state['step_count'],
                'epoch': train_iterator.epoch,
                'iter_time': t1 - iter_t0,
                'tokens': state['iter_num']
                * train.micro_batch_size
                * model.config.block_size,
                'total_tokens': (
                    state['iter_num']
                    * train.micro_batch_size
                    * model.config.block_size
                    * fabric.world_size
                ),
                'learning_rate': scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f'{val_loss:.3f}'
            fabric.print(
                f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms |"
                f" est time remaining: {humanize_seconds((TOTAL_ITERATIONS - state['iter_num']) * MOVING_AVERAGE.compute(metrics['iter_time']))}"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state['iter_num'])

        if not is_accumulating and state['step_count'] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(
                f"iter {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {'val_loss': val_loss, 'val_ppl': math.exp(val_loss)}
            fabric.log_dict(metrics, step=state['iter_num'])
            fabric.barrier()
        if (
            train.save_interval is not None
            and not is_accumulating
            and state['step_count'] % train.save_interval == 0
        ):
            checkpoint_file = (
                out_dir / f"step-{state['step_count']:06d}" / 'lit_model.pth'
            )
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f'Saving checkpoint to {str(checkpoint_file.parent)!r}')
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data[0].prompt_style, checkpoint_file.parent)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: lightning.Fabric,
    model: GPT,
    val_dataloader: DataLoader,
    tokenizer: Tokenizer,
    eval: EvalArgs,
    data: List[DataModule],
) -> torch.Tensor:
    fabric.print('Validating ...')
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch['input_ids'], batch['labels']
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(
            logits[..., :-1, :], targets[..., 1:], chunk_size=0
        )

    val_loss = losses.mean()

    # produce an example:
    instruction = (
        'Recommend a movie for me to watch during the weekend and explain the reason.'
    )
    fabric.print(instruction)

    # Simplistically take the first datamodule's prompstyle. They shouldn't be different...
    prompt = data[0].prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)

    # TODO: Our GPT block does not have the newer method of #set_kv_cache.
    # with fabric.init_tensor():
    #     # do not set `max_seq_length=max_returned_token` because memory is not a concern here
    #     model.set_kv_cache(batch_size=1)

    output = generate(
        model,
        encoded,
        max_returned_tokens=len(encoded) + eval.max_new_tokens,
        temperature=0.8,
        eos_id=tokenizer.eos_id,
    )

    # TODO: Our GPT block does not have the new method of #clear_kv_cache
    # model.clear_kv_cache()

    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )


def get_dataloaders(
    fabric: lightning.Fabric,
    data: List[DataModule],
    tokenizer: Tokenizer,
    train: TrainArgs,
    seed: int,
    num_workers=4,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = []
    val_dataset = []

    for datamodule in data:
        datamodule.connect(
            tokenizer=tokenizer,
            batch_size=train.micro_batch_size,
            max_seq_length=train.max_seq_length,
        )
        with fabric.rank_zero_first():
            fabric.print(f'Preparing [{datamodule.__repr__()}]...')
            prepared = datamodule.setup()

        fabric.print(f'Datamodule [{datamodule.repo_id}] prepared.')
        if prepared['train_dataset'] is not None:
            fabric.print(
                f'Datamodule [{datamodule.repo_id}] has a train dataset. Appended.'
            )
            train_dataset.append(prepared['train_dataset'])
        if prepared['val_dataset'] is not None:
            fabric.print(
                f'Datamodule [{datamodule.repo_id}] has a validation dataset. Appended.'
            )
            val_dataset.append(prepared['val_dataset'])

    sft_collate_fn_kwargs = {
        'ignore_index': -100,
        'pad_id': tokenizer.pad_id,
    }
    if isinstance(train.max_seq_length, int):
        sft_collate_fn_kwargs['max_seq_length'] = train.max_seq_length
    else:
        fabric.print(
            'WARN: `max_seq_length` is not defined in `TrainArgs`. It is strongly recommended to define this.'
        )

    train_dataloader = DataLoader(
        ConcatDataset(train_dataset),
        batch_size=train.micro_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        collate_fn=get_sft_collate_fn(**sft_collate_fn_kwargs),
    )

    val_dataloader = DataLoader(
        ConcatDataset(val_dataset),
        batch_size=train.micro_batch_size,
        # We want to shuffle so that if only a subset of validation is done,
        # they will at least comprise a random mix from participating datasets
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        collate_fn=get_sft_collate_fn(**sft_collate_fn_kwargs),
    )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d['input_ids']) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ['max_tokens', 'max_norm', 'tie_embeddings'])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ['epochs']), (eval, ['max_new_tokens'])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f'{__file__} requires the {name!r} argument. This is set in {args}'
                )
    if not train.epochs and not train.max_steps:
        issues.append(
            f'{__file__} requires either epochs or max_steps to be set. This is set in {train}'
        )
    if issues:
        raise ValueError('\n'.join(issues))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    from jsonargparse import CLI

    CLI(setup, as_positional=False)
