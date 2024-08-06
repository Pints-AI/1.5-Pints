from typing import TypedDict
from torch import Tensor


class SFTDatasetItem(TypedDict):
    input_ids: Tensor  # 1-dimensionality in y direction
    labels: Tensor


class SFTCollatedBatch(TypedDict):
    input_ids: Tensor  # n-dimensionality in y direction, where n is batch size
    labels: Tensor
