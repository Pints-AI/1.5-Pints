import random
from typing import List, TypedDict

import datasets
from datasets import load_dataset as load

from dpo.adapters.base import NUM_PROCESSES, DatasetAdapter, FormattedDatasetRow, Split

HUGGINGFACE_ID = 'berkeley-nest/Nectar'
SOURCES_TO_SKIP = ['anthropic-hh'] # Skip this as we want the model to be uncensored

class NectarAnswer(TypedDict):
    answer: str
    model: str
    rank: int


class NectarRow(TypedDict):
    prompt: str
    answers: List[NectarAnswer]
    source: List[str]


class Nectar(DatasetAdapter):

    splits: List[Split] = ['train']

    def __init__(self, seed: int = None):
        self.seed = seed

    def load_dataset(self, split: Split, prompt_template='chatml') -> datasets.Dataset:
        # Load huggingface dataset
        dataset = load(HUGGINGFACE_ID, split='train', num_proc=NUM_PROCESSES)

        # Drop Unused Columns
        dataset = dataset.remove_columns(
            ['turns', 'num_responses', 'good_natured']
        )

        # Print first 10 rows
        test_rows = dataset.select(range(0, 10))
        test_rows.map(format_rows_print, num_proc=NUM_PROCESSES)

        filtered_dataset = dataset.filter(filter_rows)

        # Format Dataset
        random.seed(self.seed)
        formattedDataset = filtered_dataset.map(
            format_rows(prompt_template),
            num_proc=NUM_PROCESSES
        )

        return formattedDataset



def filter_rows(row: NectarRow):
    # Remove smaples from the blacklisted sources
    for source in SOURCES_TO_SKIP:
        if source in row['source']:
            return False

    return True


def format_rows(prompt_template='chatml'):
    if prompt_template == 'chatml':
        def _format_rows(row: NectarRow) -> FormattedDatasetRow:
            chosen = row['answers'][0]['answer']
            # chosen = choose_top_answer(
            #     answers=row['answers'],
            #     filter_out_gpt=False,
            # )

            # https://arxiv.org/pdf/2310.16944
            # We construct binary preferences from UltraFeedback by selecting the
            # highest mean score as the “chosen” response and one of the remaining three at random as
            # “rejected”. We opted for random selection instead of selecting the lowest-scored response
            # to encourage diversity and make the DPO objective more challenging. As noted above, this
            # step is computed offline and does not involve any sampling from the reference model.
            rejected_pool = row['answers'][1:]
            rejected = random.choice(rejected_pool)['answer']  # Randomize the rejected answer

            # The DPOTrainer tokenization will not add a stop token. So we have to add it here ourselves.
            # and have the model learn to output the end token.
            return {
                'chosen': f'{chosen}<|im_end|>',
                'rejected': f'{rejected}<|im_end|>',
                'prompt': format_prompt(row['prompt'], 'chatml'),
            }

        return _format_rows

    raise NotImplementedError(f'Prompt template [{prompt_template}] not implemented.')


def choose_top_answer(answers: List[NectarAnswer], filter_out_gpt: bool) -> str:
    # TODO: Due to implementation of random rejected answer, this needs to be refactored and cannot be used.
    # if filter_out_gpt:
    #     # TODO: Find a way to filter out chat-gpt answers as there are rows where there are no non-chatGPT answers
    #     # If we want to filter out gpt, we will choose the top-ranked that is not from GPT model
    #     for answer in answers:
    #         if 'gpt' not in answer['model']:
    #             return answer['answer']
    #     raise NotImplementedError(
    #         'Have yet to implement a way to filter out rows where there are no non-chatGPT answers'
    #     )

    # If we want are not filtering out gpt, then we will just choose the first answer (since that will be the top-ranked one)
    return answers[0]['answer']


def format_prompt(prompt: str, promptFormat: str = 'chatml') -> str:
    assert promptFormat == 'chatml', \
        f'Error, {promptFormat} has not been implemented.'

    # Remove spaces and newlines at the start and end
    prompt = prompt.strip()

    # Human
    assert prompt.startswith('Human: ')
    prompt = prompt.removeprefix('Human: ')
    prompt = '<|im_start|>user\n' + prompt

    # Assistant
    assert prompt.endswith('Assistant:')
    prompt = prompt.removesuffix('Assistant:')
    prompt = prompt.strip()

    prompt += '<|im_end|>\n<|im_start|>assistant\n'

    return prompt


def format_rows_print(row: NectarRow) -> None:
    return
    to_print = format_rows()(row)
    print('=' * 80)
    print(to_print['prompt'])
    print('=' * 80)
