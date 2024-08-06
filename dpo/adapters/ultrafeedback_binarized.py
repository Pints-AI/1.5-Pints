from typing import List, Literal, TypedDict

from datasets import Dataset
from datasets import load_dataset as load

from dpo.adapters.base import NUM_PROCESSES, DatasetAdapter, FormattedDatasetRow, Split

HUGGINGFACE_ID = 'HuggingFaceH4/ultrafeedback_binarized'

# This is the distance between the DPO pairs before we use it. If the scores does not have a wide distance,
# it may not be useful for the model to learn, given both are just as good, or just as poor
SCORE_DISTANCE = 2.0


class UltraFeedbackItem(TypedDict):
    content: str
    role: Literal['user', 'assistant']


class UltraFeedbackRow(TypedDict):
    prompt_id: str
    prompt: str
    chosen: List[UltraFeedbackItem]
    rejected: List[UltraFeedbackItem]
    score_chosen: float
    score_rejected: float


class UltraFeedbackBinarized(DatasetAdapter):

    splits: List[Split] = ['train', 'test']

    def __init__(self, score_distance: float = SCORE_DISTANCE):
        self.score_distance = score_distance

    def load_dataset(
        self,
        split: Split,
        prompt_template='chatml'
    ) -> List[FormattedDatasetRow]:
        """
        Args:
            split: The dataset split that you want to get. Ultrafeedback has `train` and `test`.
            prompt_template: The prompt template you want to use.
            score_distance: Minimum score distance for included DPO pairs.
                I.e, 1.0 means a pair with chosen score of 9.0, and rejected score of 8.0, will be accepted (9 - 8 = 1, which is not less than 1.0)
        """

        if split.lower() == 'train':
            split = 'train_prefs'
        elif split.lower() == 'test':
            split = 'test_prefs'
        else:
            raise Exception(f"Unknown split '{split}")

        # Load dataset from Huggingface
        dataset = load(HUGGINGFACE_ID, split=split)

        # Drop unused columns
        dataset = dataset.remove_columns(['messages'])

        # Format rows
        formatted_dataset = format_dataset(dataset, prompt_template, self.score_distance)

        # Drop other unused columns post-processing
        formatted_dataset = formatted_dataset.remove_columns(
            ['prompt_id', 'score_chosen', 'score_rejected']
        )

        return formatted_dataset


def format_dataset(dataset: Dataset, prompt_template: str, score_distance: float):
    # Print first 10 rows just to check
    test_rows = dataset.select(range(0, 10))
    test_rows = test_rows.map(
        format_rows(prompt_template), num_proc=NUM_PROCESSES
    )
    print_test_rows(test_rows)

    print('Dataset length before filtering:', len(dataset))
    filtered_dataset = filter_dataset(dataset, score_distance)
    print('Dataset length after filter:', len(filtered_dataset))
    # Format Dataset
    formatted_dataset = filtered_dataset.map(
        format_rows(prompt_template), num_proc=NUM_PROCESSES
    )

    return formatted_dataset


def format_rows(prompt_template='chatml') -> FormattedDatasetRow:
    if prompt_template == 'chatml':

        def _format_row(row: UltraFeedbackRow) -> FormattedDatasetRow:
            dpo_triplets = get_dpo_triplets(row)

            formatted_prompt = f"""\
<s><|im_start|>user
{dpo_triplets['prompt']}<|im_end|>
<|im_start|>assistant
"""

            formatted_chosen = f"{dpo_triplets['chosen']}<|im_end|>"
            formatted_rejected = f"{dpo_triplets['rejected']}<|im_end|>"

            return {
                'prompt': formatted_prompt,
                'chosen': formatted_chosen,
                'rejected': formatted_rejected,
            }

        return _format_row


def get_dpo_triplets(row: UltraFeedbackRow) -> FormattedDatasetRow:
    return {
        'prompt': row['prompt'],
        'chosen': row['chosen'][1]['content'],
        'rejected': row['rejected'][1]['content'],
    }


def print_test_rows(row: UltraFeedbackRow) -> None:
    print('=' * 80)
    print(row['prompt'])
    print('=' * 80)


def filter_dataset(dataset: Dataset, score_distance: float):
    def filter_rows(score_distance: float):
        # Put all the filter criteria here
        def _filter_rows(row: UltraFeedbackRow):
            # Remove a malfeascant row about covid. It has wrong answer
            remove_prompt_ids = [
                '744aa5f9a6cbab1c168c606df3e1daf63a6cba08e20d5b8526a70627606b9f2e'
            ]
            if row['prompt_id'] in remove_prompt_ids:
                return False

            distance = row['score_chosen'] - row['score_rejected']

            if distance < score_distance:
                return False

            return True

        return _filter_rows

    return dataset.filter(filter_rows(score_distance=score_distance))
