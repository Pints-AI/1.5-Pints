from typing import Union, TypedDict
from lit_gpt.datamodules.sft_dataset_base import SFTDataset
from lit_gpt.datamodules.sft_multiturn_dataset_base import SFTMultiTurnDataset

class PreparedDataset(TypedDict):
    train_dataset: Union[SFTDataset, SFTMultiTurnDataset, None]
    val_dataset: Union[SFTDataset, SFTMultiTurnDataset, None]
    test_dataset: Union[SFTDataset, SFTMultiTurnDataset, None]
