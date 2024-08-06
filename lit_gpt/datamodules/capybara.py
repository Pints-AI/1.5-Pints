from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TypedDict
from datasets import load_dataset, Dataset

from lit_gpt.prompts import PromptStyle
from lit_gpt.datamodules.base import DataModule
from lit_gpt.datamodules.sft_multiturn_dataset_base import SFTMultiTurnDataset
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.datamodules.typings.formatted_dataset import (
    FormattedSFTDict,
    FormattedSFTMultiturnConversation,
    FormattedSFTMultiturnDataset,
)

HUGGINGFACE_ID = 'LDJnr/Capybara'
DOWNLOAD_DIR = './data/sft/capybara'
SPLIT = ['train']


class PreparedCapybara(TypedDict):
    train_dataset: SFTMultiTurnDataset
    val_dataset: None
    test_dataset: None


class CapybaraMessage(TypedDict):
    input: str
    output: str


class CapybaraRow(TypedDict):
    source: str
    conversation: List[CapybaraMessage]


@dataclass
class Capybara(DataModule):
    """Capybara data module for supervised finetuning."""

    """Whether to include multi-turn conversations in the dataset."""
    include_multiturn_conversations: bool = True
    
    """The directory in which the downloaded dataset gets saved."""
    download_dir: Path = Path(DOWNLOAD_DIR)
    
    """The repo from where the data is downloaded"""
    repo_id: str = HUGGINGFACE_ID

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTMultiTurnDataset] = field(
        default=None, init=False, repr=False
    )
    test_dataset: Optional[SFTMultiTurnDataset] = field(
        default=None, init=False, repr=False
    )

    def __init__(self, prompt_style='chatml', num_workers: Optional[int] = None):
        super().__init__()
        self.prompt_style = PromptStyle.from_name(prompt_style)
        if num_workers:
            self.num_workers = num_workers

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> List[Dataset]:
        return load_dataset(
            self.repo_id,
            split=SPLIT,
            cache_dir=self.download_dir,
            streaming=False,
        )

    def setup(self, stage: str = '') -> PreparedCapybara:
        dataset = self.prepare_data()
        train_data = format_dataset(dataset[0], self.include_multiturn_conversations)
        # TODO: Capybara doesn't have test_data. We can split it out from the train_data
        # test_data = format_dataset(dataset[1], self.include_multiturn_conversations)

        self.train_dataset = SFTMultiTurnDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        # self.test_dataset = SFTMultiTurnDataset(
        #     data=test_data,
        #     tokenizer=self.tokenizer,
        #     prompt_style=self.prompt_style,
        #     max_seq_length=self.max_seq_length,
        #     mask_prompt=self.mask_prompt,
        #     ignore_index=self.ignore_index,
        # )

        return {
            'train_dataset': self.train_dataset,
            'val_dataset': self.test_dataset,
            'test_dataset': self.test_dataset,
        }


def format_dataset(
    dataset: List[CapybaraRow], include_multi_turn_conversations: bool
) -> FormattedSFTMultiturnDataset:
    formatted: FormattedSFTMultiturnDataset = []

    for entry in dataset:
        formatted_convo: FormattedSFTMultiturnConversation = []
        convo = entry['conversation']

        for i in range(0, len(convo)):
            formatted_sft_dict: FormattedSFTDict = {
                'instruction': convo[i]['input'],
                'input': '',
                'output': convo[i]['output'],
            }
            formatted_convo.append(formatted_sft_dict)

            # If don't want to include multi turn, break after first
            # turn is appended: - no point including latter turns as
            # they become orphaned discussions without starting context
            if not include_multi_turn_conversations:
                break

        formatted.append(formatted_convo)

    return formatted
