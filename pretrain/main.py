# Support running without installing as a package
# ruff: noqa: E402
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import glob
import math
import time
from typing import Optional, Tuple, Union, Literal
import random
from os.path import isdir
from functools import partial
import torch
import lightning
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pretrain.custom_types.training_state import TrainingState, TrainingParams

# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils_old import (
    chunked_cross_entropy,
    get_default_supported_precision,
    num_parameters,
    step_csv_logger,
)
from lit_gpt.utils import CycleIterator
from lit_gpt.fused_cross_entropy import FusedCrossEntropyLoss
from lit_gpt.tokenizer import Tokenizer

TRAIN_DATA_CONFIG = [
    'Expository-Prose-V1',
    # Add other datasets if you wish
]

# In this case, validation is prepared for all datasets.
# And using 100% of it, so it's the same.
VAL_DATA_CONFIG = TRAIN_DATA_CONFIG


def setup_gpu_related_hyperparams(
    global_batch_size: int,
    gpus: int,
    micro_batch_size: int,
    warmup_steps: int,
    max_step: int,
    log_step_interval: int,
) -> Tuple[dict[str, int], int]:
    batch_size = global_batch_size // gpus
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0
    warmup_iters = warmup_steps * gradient_accumulation_steps

    max_iters = max_step * gradient_accumulation_steps
    log_iter_interval = log_step_interval * gradient_accumulation_steps

    gpu_hyperparams = {
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'warmup_iters': warmup_iters,
        'max_iters': max_iters,
        'lr_decay_iters': max_iters,
        'log_iter_interval': log_iter_interval,
        'devices': gpus,
        'global_batch_size': global_batch_size,
        'micro_batch_size': micro_batch_size,
        'max_step': max_step,
        'warmup_steps': warmup_steps,
        'log_step_interval': log_step_interval,
    }

    return gpu_hyperparams, log_iter_interval


def setup_paths(
    model_name: str, data_dir: Union[Path, None], out_dir: Union[Path, None]
) -> dict[str, Path]:
    """
    Set up and resolve paths for model data directory, output directory.
    """

    if data_dir is not None:
        data_dir = data_dir.resolve()

    assert (
        data_dir is not None and data_dir.exists()
    ), f'`data_dir` of [{data_dir}] not a valid directory!'

    if out_dir is None:
        out_dir = Path('models') / model_name.lower()

    path_dict = {
        'data_dir': data_dir,
        'out_dir': out_dir,
        'model_name': model_name,
    }

    return path_dict


def setup_resume_checkpoint(
    out_dir: Union[Path, None],
    resume: Union[bool, Path],
) -> Path | Literal[False]:
    """
    Set up and resolve paths checkpoint.
    """
    if resume is True:
        # this mean we find the latest checkpoint and resume from there
        resume = sorted(out_dir.glob('**/*.pth'))[-1]
        resume = resume.resolve()

    elif isinstance(resume, Path):
        # we just check if the checkpoint provided is correct
        assert (
            resume is not None and resume.exists()
        ), f'checkpoint_path[{resume}] is not a valid directory!'
        resume = resume.resolve()

    if resume is not False:
        print(f'Resuming training from checkpoint: [{resume}]')

    return resume


def setup_strategy(gpus: int, tpu: bool) -> Union[FSDPStrategy, XLAStrategy]:
    """
    Set up the strategy. In future, we can use this to set other types of strategies, such as deepspeed.
    """
    print('Setting strategy...')
    if gpus > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            gpus = 'auto'
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type='full',
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = 'auto'

    return strategy


def setup_ignore_index(
    tokenizer_dir: Union[Path, None],
) -> int:
    """
    Ignore index should ideally be the index of the padding token. Most foundational models does not come with one.
    This function tries to get it, but defaults back to -100 (default pytorch ignore index) if not found.
    """
    print('Setting up ignore index...')
    # Attempt to get tokenizer pad token for ignore_index later
    pad_token_id = None
    pad_token = None
    ignore_index = -100
    if tokenizer_dir is None:
        message = (
            'WARN: You have not specified the tokenizer used to prepare your data.'
        )
        message += (
            ' We will not know what is the `pad_token_id` to ignore cross entropy loss.'
        )
        print(message)
    else:
        tokenizer = Tokenizer(tokenizer_dir)
        pad_token_id = tokenizer.pad_id

        # Now check pad token.
        if not isinstance(pad_token_id, int):
            message = f'WARN: Invalid `pad_token_id` of "{pad_token_id}" found.'
            message += ' It cannot be used to ignore cross entropy loss.'
            print(message)

        print('Setting pad token...')
        # Sometimes it has an index, but cannot be decoded.
        try:
            pad_token = tokenizer.decode(torch.IntTensor([tokenizer.pad_id]), False)
        except Exception:
            message = 'WARN: Invalid `pad_token`, it cannot be decoded.'
            message += ' It cannot be used to ignore cross entropy loss.'
            print(message)

        if pad_token:
            print(
                f'pad is "{pad_token}" with id of "{pad_token_id}". This will be use as `ignore_index`.'
            )
            # Pretrain data will not have ignore_index.
            # Instead it will be padded up to uniform sequence lengths
            # But we don't want model to predict these, so pad_token_id becomes the ignore_index
            ignore_index = pad_token_id
        else:
            print(
                'WARN: There is no pad token found. Models trained with dataset padded with other tokens may not work.'
            )

    return ignore_index


def setup(
    data_dir: Path,
    out_dir: Optional[Path] = None,
    gpus: int = 8,
    global_batch_size=512,
    learning_rate=4e-4,
    micro_batch_size=8,
    max_step=48090
    * 2,  # changes are made to global_batch_size, or dataset, this needs to change.
    warmup_steps=2000,
    log_step_interval=10,
    eval_iters=100,
    save_step_interval=4000,
    eval_step_interval=4000,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    min_lr=4e-5,
    precision: _PRECISION_INPUT = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    model_name='1.5-Pints-2k',
    wandb_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    tokenizer_dir: Optional[Path] = None,
) -> None:
    """
    Sets up the environment and configurations for training a model.

    Args:
        data_dir (Optional[Path]): Directory containing the training data. Defaults to None.
        out_dir (Optional[Path]): Directory to save the outputs. Defaults `models/{model_name}`.
        gpus (int): Number of GPUs to use. Defaults to 8. GOTCHA: Somehow naming this `devices` will be ignored as a CLI param.
        global_batch_size (int): This is the training batch size. A larger batch size accumulates more gradients and can be faster. A smaller batch size reduces parallelism, but improves stability. See Llama 3.1 paper https://arxiv.org/pdf/2407.21783 (3.4.1 Initial Pre-Training). Defaults to 512.
        learning_rate (float): This is the max learning rate for the optimizer. Defaults to 4e-4.
        micro_batch_size (int): Batch size for each device. Increase this to the memory limit of each GPU for faster training. Defaults to 8.
        max_step (int): Maximum number of training steps, so that the optimizer can schedule the learning rate. Defaults to 48090 * 2.
        warmup_steps (int): Number of steps for learning rate warmup. Defaults to 2000.
        log_step_interval (int): Interval for logging training progress. Defaults to 10.
        eval_iters (int): Number of iterations for evaluation. Defaults to 100.
        save_step_interval (int): Interval for saving the model checkpoints. Defaults to 4000.
        eval_step_interval (int): Interval for evaluation during training. Defaults to 4000.
        weight_decay (float): Weight decay for regularization. Defaults to 0.1.
        beta1 (float): Beta1 parameter for the Adam optimizer. Defaults to 0.9.
        beta2 (float): Beta2 parameter for the Adam optimizer. Defaults to 0.95.
        grad_clip (float): Gradient clipping value. Most people use 1.0. Defaults to 1.0.
        decay_lr (bool): Whether to decay the learning rate during training. Defaults to True.
        min_lr (float): Minimum learning rate if decay_lr is True. Defaults to 4e-5.
        precision (Optional[Literal['32-true', 'bf16-mixed', 'bf16-true', '16-mixed', '16-true']]): Precision type for training. Defaults to None.
        tpu (bool): Whether to use TPU for training. Defaults to False.
        resume (Union[bool, Path]): Whether to resume training from a checkpoint. If a Path is provided, training will resume from the specified checkpoint. Defaults to False.
        model_name (str): Name of the model. Used for organizing outputs and logging. Defaults to '1.5-Pints-2k'.
        wandb_name (Optional[str]): Name for the Weights and Biases run. Defaults to None.
        wandb_project (Optional[str]): Project name for Weights and Biases. Defaults to None.
        tokenizer_dir (Optional[Path]): Directory containing the tokenizer. Must match the tokenizer used for preparing the data. Defaults to None.
        checkpoint_path (Optional[Path]): Path to the checkpoint directory for continuing training. Defaults to None.
    """

    print('Initializing parameters...')

    # put all them all inside the dict for passing it around
    training_params: TrainingParams = {
        'learning_rate': learning_rate,
        'eval_iters': eval_iters,
        'save_step_interval': save_step_interval,
        'eval_step_interval': eval_step_interval,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'grad_clip': grad_clip,
        'decay_lr': decay_lr,
        'min_lr': min_lr,
        'out_dir': out_dir,
    }

    # Setup Data Paths
    path_dict = setup_paths(model_name, data_dir, out_dir)
    training_params.update(path_dict)

    resume = setup_resume_checkpoint(path_dict['out_dir'], resume)

    # Hyper params that depends on `gpus`
    gpu_hyperparams, log_iter_interval = setup_gpu_related_hyperparams(
        global_batch_size,
        gpus,
        micro_batch_size,
        warmup_steps,
        max_step,
        log_step_interval,
    )
    training_params.update(gpu_hyperparams)

    # Get the precision
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    # Setup Strategy
    strategy = setup_strategy(gpus, tpu)

    print('Initializing CSV and Wandb loggers...')
    wandb_logger = WandbLogger(name=wandb_name, project=wandb_project)
    logger = step_csv_logger(
        'out', model_name.lower(), flush_logs_every_n_steps=log_iter_interval
    )

    # Setup ignore_index (try to get and use padding token index)
    ignore_index = setup_ignore_index(tokenizer_dir)

    fabric = lightning.Fabric(
        devices=gpus,
        strategy=strategy,
        precision=precision,
        loggers=[logger, wandb_logger],
    )

    fabric.print('Running pretrain with parameters:\n\n', training_params)
    # fabric.launch(main, train_data_dir, val_data_dir, resume)

    main(fabric, data_dir, resume, training_params, ignore_index=ignore_index)


def main(
    fabric: lightning.Fabric,
    data_dir: Path,
    resume: Union[Path, Literal[False]],
    training_params: TrainingParams,
    ignore_index=-100,
):
    monitor = Monitor(
        fabric,
        window_size=2,
        time_unit='seconds',
        log_iter_interval=training_params['log_iter_interval'],
    )

    # Create model out folder only for the first node.
    if fabric.global_rank == 0:
        training_params['out_dir'].mkdir(parents=True, exist_ok=True)

    config = Config.from_name(training_params['model_name'])

    train_dataloader, val_dataloader = create_dataloaders(
        break_into_chunks=training_params['devices'],
        batch_size=training_params['micro_batch_size'],
        block_size=config.block_size,
        fabric=fabric,
        data_dir=data_dir,
        seed=3407,
    )

    if train_dataloader is not None:
        fabric.print('Train dataloader prepared successfully')

    if val_dataloader is None:
        fabric.print('Setting up fabric dataloader for training only...')
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        fabric.print('Validation dataloader prepared successfully.')
        fabric.print('Setting up fabric dataloaders for training and validation...')
        train_dataloader, val_dataloader = fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )

    # same seed for every process to init model (FSDP)
    fabric.seed_everything(3407)

    fabric.print(f'Loading model with {config.__dict__}')

    t0 = time.perf_counter()

    with fabric.init_module(empty_init=False):
        model = GPT(config)
        if (
            resume is False
        ):  # this means we are training from scratch, hence need to initalize the weights for all model layers
            model.apply(partial(model._init_weights, n_layer=config.n_layer))

    instantiation_time = time.perf_counter() - t0

    fabric.print(f'Model instantiated in: {instantiation_time:.02f} seconds.')
    fabric.print(f'Total parameters {num_parameters(model):,}')

    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        betas=(training_params['beta1'], training_params['beta2']),
        foreach=False,
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state: TrainingState = {
        'model': model,
        'optimizer': optimizer,
        'iter_num': 0,
        'step_count': 0,
        'epoch': 0,
    }

    if isinstance(
        resume, Path
    ):  # checkpoint path is provided, hence load the checkpoint from the directory
        fabric.print(f'Loading from checkpoint {resume}')
        fabric.load(resume, state, strict=True)

    train_time = time.perf_counter()
    train(
        fabric,
        state,
        train_dataloader,
        val_dataloader,
        monitor,
        resume,
        training_params,
        ignore_index=ignore_index,
    )
    fabric.print(f'Training time: {(time.perf_counter()-train_time):.2f}s')
    if fabric.device.type == 'cuda':
        mem_used = torch.cuda.max_memory_allocated() / 1024**3
        fabric.print(f'Memory used: {mem_used:.02f} GB')


def train(
    fabric: lightning.Fabric,
    state: TrainingState,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    monitor: Monitor,
    resume: Union[Path, bool],
    training_params: TrainingParams,
    ignore_index=-100,
):
    """Trains the model"""

    model = state['model']
    optimizer = state['optimizer']

    if val_dataloader is not None:
        validate(
            fabric, model, val_dataloader, training_params['eval_iters']
        )  # sanity check

    with torch.device('meta'):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = (
            estimate_flops(meta_model) * training_params['micro_batch_size']
        )

        fabric.print(
            f'Estimated TFLOPs: { estimated_flops * fabric.world_size / 1e12:.2f}'
        )
        x = torch.randint(
            0, 1, (training_params['micro_batch_size'], model.config.block_size)
        )
        # measured_flos run in meta. Will trigger fusedRMSNorm error
        # measured_flops = measure_flops(meta_model, x)
        # fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == 'xla':
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    initial_iter = state['iter_num']
    curr_iter = 0

    loss_func = FusedCrossEntropyLoss(ignore_index=ignore_index)
    train_dataloader: Union[CycleIterator, DataLoader] = CycleIterator(train_dataloader)

    for train_data in train_dataloader:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue

            resume = False
            curr_iter = -1
            fabric.barrier()
            resume_time_taken = time.perf_counter() - total_t0
            fabric.print(f'resume finished, taken {resume_time_taken} seconds')

        if state['iter_num'] >= training_params['max_iters']:
            break

        # determine and set the learning rate for this iteration
        lr = (
            get_lr(state['iter_num'], training_params)
            if training_params['decay_lr']
            else training_params['learning_rate']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        targets = train_data[:, 1 : model.config.block_size + 1].contiguous()

        # TODO: We should consider fixing the dataset preparation code. This is a temp solution to skip over fully-padding sequences.
        #       The reason why we have fully padded sequence is because when we pack the data into file, each file represents a big
        #       chunk of tokens, which is Desired Sequence Length * N ("Big Chunk"), where N can be any arbitrary number, only to
        #       constrain the file size. When coming to the end of packing, where the remaining data is not sufficient to fill up the
        #       Big Chunk, we will end up with a lot of padding at the end, which when we iterate, will result in plucking out fully
        #       padded sequences. I.e.:
        #
        #       A Big Chunk that doesn't have enough data to be fully packed looks like this:
        #
        #       [ 1, 2, 3, 4, ................<EOS>, <pad_id>, <pad_id>, ....... <pad_id> ]
        #               Suppose we pluck out a sequence from here ^   to here ^
        #
        #               We end up with a fully padded sequence of [<pad_id>.... <pad_id>]
        #
        #       The solution is to fix DatasetPreparer, to look through all the last files again after packing is complete and
        #       repack those, to result in only 1 file that may contain a lot of padding.
        if torch.all(input_ids == ignore_index):
            fabric.print('Skipping over empty data.')
            continue

        is_accumulating = (state['iter_num'] + 1) % training_params[
            'gradient_accumulation_steps'
        ] != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = loss_func(logits, targets)
            # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / training_params['gradient_accumulation_steps'])

        if not is_accumulating:
            fabric.clip_gradients(
                model, optimizer, max_norm=training_params['grad_clip']
            )
            optimizer.step()
            optimizer.zero_grad()
            state['step_count'] += 1

        elif fabric.device.type == 'xla':
            xm.mark_step()

        state['iter_num'] += 1
        state['epoch'] = train_dataloader.epoch
        # input_id: B L
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"epoch {state['epoch']} iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (training_params['max_iters'] - state['iter_num']) / 3600:.2f} hours. "
            # print days as well
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (training_params['max_iters'] - state['iter_num']) / 3600 / 24:.2f} days. "
        )

        monitor.on_train_batch_end(
            state['iter_num'] * training_params['micro_batch_size'],
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            state['step_count'],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item(),
        )

        if (
            val_dataloader is not None
            and not is_accumulating
            and state['step_count'] % training_params['eval_step_interval'] == 0
        ):
            t0 = time.perf_counter()
            val_loss = validate(
                fabric, model, val_dataloader, training_params['eval_iters']
            )
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(
                f'step {state["iter_num"]}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms'
            )
            fabric.log_dict(
                {
                    'metric/val_loss': val_loss.item(),
                    'total_tokens': model.config.block_size
                    * (state['iter_num'] + 1)
                    * training_params['micro_batch_size']
                    * fabric.world_size,
                },
                state['step_count'],
            )
            fabric.log_dict(
                {
                    'metric/val_ppl': math.exp(val_loss.item()),
                    'total_tokens': model.config.block_size
                    * (state['iter_num'] + 1)
                    * training_params['micro_batch_size']
                    * fabric.world_size,
                },
                state['step_count'],
            )
            fabric.barrier()

        save_checkpoint(
            fabric,
            is_accumulating=is_accumulating,
            state=state,
            save_step_interval=training_params['save_step_interval'],
            out_dir=training_params['out_dir'],
        )

    save_checkpoint(
        fabric,
        is_accumulating=False,  # Doesn't matter
        state=state,
        save_step_interval=training_params['save_step_interval'],
        out_dir=training_params['out_dir'],
        is_last=True,
    )


@torch.no_grad()
def validate(
    fabric: lightning.Fabric,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    eval_iters: int,
) -> torch.Tensor:
    """Run validation and calculate loss."""

    fabric.print('Validating ...')
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous()
        targets = val_data[:, 1 : model.config.block_size + 1].contiguous()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)

        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()

    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    break_into_chunks: int,
    batch_size: int,
    block_size: int,
    data_dir: Path,
    fabric,
    shuffle: bool = True,
    seed: int = 12345,
    split='train',
) -> DataLoader:
    """Create a dataloader"""

    data_config = TRAIN_DATA_CONFIG if split == 'train' else VAL_DATA_CONFIG

    # Support training without validation split.
    if not VAL_DATA_CONFIG or len(VAL_DATA_CONFIG) < 1:
        return None

    consolidated_filenames = []

    for dataset_name in data_config:
        dataset_dir = data_dir / dataset_name / split

        if not isdir(dataset_dir):
            if split == 'train':
                error_msg = f'Directory [{dataset_dir}] for dataset [{dataset_name}] does not exist.'
                error_msg += f' Please check. Configuration:\n\n{TRAIN_DATA_CONFIG}'
                raise RuntimeError(error_msg)
            else:
                error_msg = f'Validation folder [{dataset_dir}] missing for dataset[{dataset_name}].'
                error_msg += 'Did you misconfigure? Or forget to include? Configuration:\n\n{VAL_DATA_CONFIG}'
                fabric.print(error_msg)

        # In our data preparation, files are prefixed as such
        prefix = f'{split}_{dataset_name}'
        filepath_pattern = f'{dataset_dir}/{prefix}*'

        filenames = sorted(glob.glob(filepath_pattern))
        number_files = len(filenames)
        if number_files < 1:
            raise RuntimeError(
                f'No data found at "{filepath_pattern}". Did you specify the right directory?'
            )
        fabric.print(f'Found [{number_files}] files in "{filepath_pattern}".')

        consolidated_filenames.extend(filenames)

    random.seed(seed)
    random.shuffle(consolidated_filenames)
    dataset = PackedDataset(
        consolidated_filenames,
        # n_chunks control the buffer size.
        # Note that the buffer size also impacts the random shuffle
        # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
        n_chunks=break_into_chunks,
        block_size=block_size,
        shuffle=shuffle,
        seed=seed + fabric.global_rank,
        num_processes=fabric.world_size,
        process_rank=fabric.global_rank,
    )

    fabric.print(
        f'Combined data for [{split}] dataset. Total [{len(consolidated_filenames)}] files.'
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    break_into_chunks: int,
    batch_size: int,
    block_size: int,
    fabric,
    data_dir: Path,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    """This calls `create_dataloader` twice, one for train, another for validation."""

    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        break_into_chunks=break_into_chunks,
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=data_dir,
        shuffle=True,
        seed=seed,
        split='train',
    )

    val_dataloader = create_dataloader(
        break_into_chunks=break_into_chunks,
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=data_dir,
        shuffle=False,
        seed=seed,
        split='validation',
    )

    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, training_params: TrainingParams):
    """Get learning rate"""

    # 1) linear warmup for warmup_iters steps
    if it < training_params['warmup_iters']:
        return training_params['learning_rate'] * it / training_params['warmup_iters']

    # 2) if it > lr_decay_iters, return min learning rate
    if it > training_params['lr_decay_iters']:
        return training_params['min_lr']

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - training_params['warmup_iters']) / (
        training_params['lr_decay_iters'] - training_params['warmup_iters']
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return training_params['min_lr'] + coeff * (
        training_params['learning_rate'] - training_params['min_lr']
    )


def save_checkpoint(
    fabric: lightning.Fabric,
    is_accumulating: bool,
    state: TrainingState,
    save_step_interval: int,
    out_dir: Path,
    is_last: bool = False,
):
    """WARNING: This function CANNOT NOT be called only on a single GPU device, it MUST BE called on ALL gpus. This
    is because fabric.save() calls

    ```
    if fabric.global_rank == 0:
        torch.save(state, filename)
    fabric.barrier()
    ```

    under the hood. This was found from https://lightning.ai/forums/t/question-about-how-fabric-models-are-saved-loaded/3045/2

    And according to the docs for fabric.barrier(), it blocks until all processes in the group have
    responded. Thus, fabric.barrier() MUST BE called on ALL gpus, otherwise the program will hang indefinitely.
    """
    checkpoint_file = out_dir / f"step-{state['step_count']:08d}" / 'lit_model.pth'
    if is_last:
        # Attempt to mkdir the directory first, other will error out
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        fabric.print(f'Saving final checkpoint to {str(checkpoint_file)!r}')
        fabric.save(checkpoint_file, state)
        fabric.print('Finished saving final checkpoint')

    # When it's no longer accumulating the, we can do a save based on the save step interval
    arrived_at_save_step = state['step_count'] % save_step_interval == 0
    if not is_accumulating and arrived_at_save_step:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        fabric.print(f'Saving checkpoint to {str(checkpoint_file)!r}')
        fabric.save(checkpoint_file, state)


if __name__ == '__main__':
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision('high')

    from jsonargparse import CLI

    CLI(setup, as_positional=False)
