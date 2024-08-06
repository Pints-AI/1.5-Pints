from typing import List, Literal, TypedDict

from datasets import Dataset
from datasets import load_dataset as load

from dpo.adapters.base import NUM_PROCESSES, DatasetAdapter, FormattedDatasetRow, Split

HUGGINGFACE_ID = 'argilla/distilabel-capybara-dpo-7k-binarized'


class DistilabelCapybaraDPOItem(TypedDict):
    content: str
    role: Literal['user', 'assistant']


class DistilabelCapybaraDPORow(TypedDict):
    source: str
    category: str
    prompt: str
    candidates_completions: List[str]
    candidate_policies: List[str]
    ranks: List[int]
    rank_str: str
    chosen_policy: str
    chosen: List[DistilabelCapybaraDPOItem]
    rejected_policy: str
    rejected: List[DistilabelCapybaraDPOItem]


class DistilabelCapybaraDPO(DatasetAdapter):

    splits: List[Split] = ['train']

    # TODO: Refactor prompt_template, use tokenizer.apply_chat_template in main.py instead.
    def load_dataset(self, split: Split, prompt_template='chatml') -> List[FormattedDatasetRow]:
        """
        Args:
            prompt_template: The prompt template you want to use.
        """

        # Load dataset from Huggingface
        dataset = load(HUGGINGFACE_ID, split='train')

        # Format rows
        formatted_dataset = format_dataset(dataset, prompt_template)

        return formatted_dataset


def format_dataset(dataset: Dataset, prompt_template: str):
    # Print first 10 rows just to check
    test_rows = dataset.select(range(0, 10))
    test_rows = test_rows.map(
        format_rows(prompt_template), num_proc=NUM_PROCESSES
    )
    print_test_rows(test_rows)

    print('Dataset length:', len(dataset))
    # Format Dataset
    formatted_dataset = dataset.map(
        format_rows(prompt_template), num_proc=NUM_PROCESSES
    )

    return formatted_dataset


def format_rows(prompt_template='chatml') -> FormattedDatasetRow:
    if prompt_template != 'chatml':
        raise ValueError(f'prompt_template "{prompt_template}" not supported.')

    def _format_row(row: DistilabelCapybaraDPORow) -> FormattedDatasetRow:
        assert row['chosen'][0]['content'] == row['rejected'][0]['content']

        formatted_prompt = f"""<s><|im_start|>user
{row['chosen'][0]['content']}<|im_end|>
<|im_start|>assistant
"""

        formatted_chosen = f"{row['chosen'][1]['content']}<|im_end|>"
        formatted_rejected = f"{row['rejected'][1]['content']}<|im_end|>"

        return {
            'prompt': formatted_prompt,
            'chosen': formatted_chosen,
            'rejected': formatted_rejected,
        }

    return _format_row


def print_test_rows(row: DistilabelCapybaraDPORow) -> None:
    print('=' * 80)
    print(row['prompt'])
    print('=' * 80)
