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

HUGGINGFACE_ID = 'togethercomputer/llama-instruct'
DOWNLOAD_DIR = './data/sft/llama-instruct'
SPLIT = ['train']

class PreparedLlamaInstruct(TypedDict):
    train_dataset: SFTMultiTurnDataset
    val_dataset: None
    test_dataset: None


class LlamaInstructRow(TypedDict):
    text: str


@dataclass
class LlamaInstruct(DataModule):
    """LlamaInstruct data module for supervised finetuning."""

    """Whether to include multi-turn conversations in the dataset."""
    include_multiturn_conversations: bool = False
    
    """The directory in which the downloaded dataset gets saved."""
    download_dir: Path = Path(DOWNLOAD_DIR)
    
    """The repo from where the data is downloaded"""
    repo_id: str = HUGGINGFACE_ID

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    # `batch_size` has no use anymore, as the dataloaders are moved outside
    # in order to support training with multiple SFT datasets at one go.
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTMultiTurnDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTMultiTurnDataset] = field(default=None, init=False, repr=False)

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

    def setup(self, stage: str = '') -> PreparedLlamaInstruct:
        dataset = self.prepare_data()

        train_data = format_dataset(dataset[0], self.include_multiturn_conversations)
        # TODO: WizardLMEvolInstructV2 doesn't have test_data. We can split it out from the train_data
        # test_data = format_dataset(dataset[1], self.include_multiturn_conversations)

        self.train_dataset = SFTMultiTurnDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

        return {
            'train_dataset': self.train_dataset,
            'val_dataset': self.test_dataset,
            'test_dataset': self.test_dataset,
        }


def format_dataset(
    dataset: List[LlamaInstructRow],
    include_multi_turn_conversations=bool,
) -> FormattedSFTMultiturnDataset:
    formatted: FormattedSFTMultiturnDataset = []

    for entry in dataset:
        conversation = entry["text"]

        # It came with [INST] format. Turn all of it into a special token and split one time.
        split_turns = conversation.replace('[INST]', '<||SPLIT||>').replace('[/INST]', '<||SPLIT||>').split('<||SPLIT||>')

        # Strip the content of any spaces, turn into a flat list
        # `cleaned_split_turns` will contain pairs of alternating user and assistant content
        cleaned_split_turns: List[str] = []
        for turn in split_turns:
            turn = turn.strip()
            if len(turn) < 1:
                continue

            cleaned_split_turns.append(turn)

        formatted_convo: FormattedSFTMultiturnConversation = []
        for i in range(0, len(cleaned_split_turns) - 1, 2):
            formatted_sft_dict: FormattedSFTDict = {
                'instruction': cleaned_split_turns[i],
                'input': '',
                'output': cleaned_split_turns[i+1]
            }
            formatted_convo.append(formatted_sft_dict)

            # If don't want to include multi turn, break after first
            # turn is appended: - no point including latter turns as
            # they become orphaned discussions without starting context
            if not include_multi_turn_conversations:
                break
            

        formatted.append(formatted_convo)
            
    return formatted
