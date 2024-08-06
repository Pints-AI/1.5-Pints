from typing import TypedDict
from pathlib import Path

class TrainingParams(TypedDict):
    devices: int
    batch_size: int
    gradient_accumulation_steps: int
    warmup_iters: int
    max_iters: int
    lr_decay_iters: int
    log_iter_interval: int
    model_name: str
    out_dir: Path
    global_batch_size: int
    learning_rate: float
    micro_batch_size: int
    max_step: int
    warmup_steps: int
    log_step_interval: int
    eval_iters: int
    eval_step_interval: int
    save_step_interval: int
    weight_decay: float
    beta1: int
    beta2: int
    grad_clip: int
    decay_lr: bool
    min_lr: float
    