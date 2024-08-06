from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, TypedDict, Literal
from datasets import load_dataset, Dataset

from lit_gpt.prompts import PromptStyle
from lit_gpt.datamodules.base import DataModule
from lit_gpt.datamodules.sft_dataset_base import SFTDataset
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.datamodules.typings.formatted_dataset import (
    FormattedSFTSingleturnConversation,
    FormattedSFTSingleturnDataset,
)

DATA_FILES_PATH = '.data/sft/slim-orca-idontknow/slim-orca-idontknow.parquet'
DOWNLOAD_DIR = './data/sft/slim-orca-idontknow'
SPLIT = ['train']

class PreparedSlimOrcaIDK(TypedDict):
    train_dataset: SFTDataset
    val_dataset: None
    test_dataset: None


# Use declarative syntax as 'from' will not be allowed using class syntax
OpenOrcaConversationTurn = TypedDict(
    'OpenOrcaConversation', {'from': Literal['system', 'human', 'gpt'], 'value': str}
)


class SlimOrcaIDKRow(TypedDict):
    conversations: List[OpenOrcaConversationTurn]


@dataclass
class SlimOrcaIDK(DataModule):
    """SlimOrcaIDK data module for supervised finetuning."""

    """Whether to include multi-turn conversations in the dataset."""
    include_multiturn_conversations: bool = False
    
    """The directory in which the downloaded dataset gets saved."""
    download_dir: Path = Path(DOWNLOAD_DIR)
    
    """The repo from where the data is downloaded"""
    repo_id: str = DATA_FILES_PATH
    
    data_files_path: str = DATA_FILES_PATH

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    test_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __init__(
        self,
        prompt_style='chatml',
        data_files_path='data/slim_orca_idontknow/slim-orca-idontknow.parquet',
    ):
        super().__init__()
        self.prompt_style = PromptStyle.from_name(prompt_style)
        self.data_files_path = self.repo_id = data_files_path

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
            'parquet',
            split=SPLIT,
            data_files=self.repo_id,
            streaming=False,
        )

    def setup(self, stage: str = '') -> PreparedSlimOrcaIDK:
        dataset = self.prepare_data()
        train_data = format_dataset(dataset[0], self.include_multiturn_conversations)
        # TODO: SlimOrcaIDK doesn't have test_data. We can split it out from the train_data
        # test_data = format_dataset(dataset[1], self.include_multiturn_conversations)

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        # self.test_dataset = SFTDataset(
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
    dataset: List[SlimOrcaIDKRow],
    # `include_multi_turn_conversations` kept for backward compatibility with litgpt
    include_multi_turn_conversations: False,
) -> FormattedSFTSingleturnDataset:
    formatted: FormattedSFTSingleturnDataset = []

    for entry in dataset:
        conversation = entry['conversations']

        # NOTE: We are not training with system message. This is a deliberate decision.
        #       We find that system messages generally weaken the model, and usually can be part of the instruction.
        #       Additionally, the current litgpt instruction/input/output doesn't cater for system messages

        # system_message = conversation[0]
        # if system_message['from'] != 'system':
        #     print(
        #         f'WARN: A Slim orca row is corrupted. Expected role to be `user`, but is `{conversation[i]["from"]}` instead.'
        #     )

        # Start from index 1, which should be human message
        human_message = conversation[1]
        if human_message['from'] != 'human':
            print(
                f'WARN: A Slim orca row is corrupted. Expected role to be `user`, but is `{human_message["from"]}` instead.'
            )

        ai_message = conversation[2]
        if ai_message['from'] != 'gpt':
            print(
                f'WARN: A Slim orca row  is corrupted. Expected role to be `assistant`, but is `{ai_message["from"]}` instead.'
            )

        formatted_sft_dict: FormattedSFTSingleturnConversation = {
            'instruction': human_message['value'],
            'input': '',
            'output': ai_message['value'],
        }

        formatted.append(formatted_sft_dict)

    return formatted
