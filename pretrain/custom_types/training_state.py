from typing import TypedDict, Any, Union, Tuple
from lightning.fabric.wrappers import _FabricOptimizer
from .training_params import TrainingParams


class TrainingState(TypedDict):
    model: Any  # fabric doesn't type this...
    optimizer: Union[_FabricOptimizer, Tuple[_FabricOptimizer, ...]]
    iter_num: int
    step_count: int
    epoch: int
