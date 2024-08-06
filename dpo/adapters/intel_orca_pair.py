from typing import List

from datasets import load_dataset as load

from dpo.adapters.base import DatasetAdapter, FormattedDatasetRow, Split

# TODO: Not ready for use, need to add in chat template formatting

HUGGINGFACE_ID = 'Intel/orca_dpo_pairs'


class OrcaDPOPairs(DatasetAdapter):

    splits: List[Split] = ['train']

    def load_dataset(self, split: Split) -> List[FormattedDatasetRow]:
        # Load huggingface dataset
        dataset = load(HUGGINGFACE_ID, split='train')

        # Remove unused column
        dataset = dataset.remove_columns(['system'])

        # Rename columns to names required by dpo trainer
        dataset.rename_column('question', 'prompt')

        return dataset
