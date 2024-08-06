# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from functools import partial
from typing import List, Optional, Union
from multiprocessing import cpu_count

import torch
from lightning import LightningDataModule

from lit_gpt.tokenizer import Tokenizer
from lit_gpt.prompts import PromptStyle
from lit_gpt.datamodules.typings.base import SFTDatasetItem, SFTCollatedBatch
from lit_gpt.datamodules.typings.prepared_dataset import PreparedDataset


class DataModule(LightningDataModule):
    """Base class for all data modules in LitGPT."""

    """Whether to mask the prompt section from the label (with ``ignore_index``)."""
    mask_prompt: bool = True
    """The style to apply to instruction prompts. See `litgpt.prompts` for a list of available styles."""
    prompt_style: Union[str, PromptStyle] = 'chatml'
    """The index to use for elements to be ignored in the label."""
    ignore_index: int = -100
    """The random seed for shuffling the dataset."""
    seed: int = 42
    """How many DataLoader processes to use for loading."""
    num_workers = max(1, cpu_count() // 2) # Use half of available cores
    """The directory in which the downloaded dataset gets saved."""
    include_multiturn_conversations: bool = True
    """The repo from where the data is downloaded"""
    repo_id: str = ''

    @abstractmethod
    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        """All settings that can't be determined at the time of instantiation need to be passed through here
        before any dataloaders can be accessed.
        """

    def setup(self, stage: str = '') -> PreparedDataset:
        # Stub is to redefine the default signature, because the concept of 'stage' does not exist in LitGPT
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def get_sft_collate_fn(
    max_seq_length: int = -1,
    pad_id: int = 0,
    ignore_index: int = -100,
    eos_id: Optional[int] = 2,
):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    Args:
        max_seq_length: Anything more than this will be truncated.
        pad_id: This defaults to -100. It is not recommend to use the tokenizer's pad_id here.
            We need to ignore cross-entropy loss on the pad tokens, but if pad_id is provided, it will not be ignored.
        ignore_index: Defaults to -100 as pytorch's default.
        eos_id: Specify if you wish to use other eos_id, otherwise it defaults to 2. This will be added to the end
                if `max_seq_length` is reached. Otherwise, you can provied `None` to not add.
    """

    if pad_id == 0:
        print(
            'WARN: Pad token is 0, which is usually <unk> token. It is more correct to create and use a pad token.'
        )
    if pad_id is None:
        pad_id = 0
        print("WARN: datamodules/base.py: pad_id was None, changed to 0")

    return partial(
        _sft_collate_fn,
        max_seq_length=max_seq_length,
        pad_id=pad_id,
        ignore_index=ignore_index,
        eos_id=eos_id,
    )


def _sft_collate_fn(
    samples: List[SFTDatasetItem],
    max_seq_length: int,
    pad_id: int,
    ignore_index: int,
    eos_id: Optional[int],
) -> SFTCollatedBatch:
    batched = {}
    for key in ('input_ids', 'labels'):
        # So we want to pad the inputs only.
        # For the labels, we don't want the model to learn to predict the pad tokens
        # So we just pad it with ignore_index
        pad_value = pad_id if key == 'input_ids' else ignore_index

        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples],
            batch_first=True,
            padding_value=pad_value,

        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

            if isinstance(eos_id, int):
                # Replace last token with eos_id only if it's not `ignore_index` or `pad_id`:
                condition = (batched[key][:, -1] != ignore_index) & (
                    batched[key][:, -1] != pad_id
                )
                batched[key][condition, -1] = eos_id

            elif eos_id is None:
                pass
            else:
                raise TypeError(
                    f'`eos_id` of value[{eos_id}] and type[{type(eos_id)}] is not valid.'
                )

    return batched
