from typing import List, Tuple

from datasets import Dataset, concatenate_datasets

from dpo.adapters.base import DatasetAdapter, FormattedDatasetRow

# The amount of test data to split from datasets that doesn't have
# test/eval sets
TEST_SIZE = 0.1


def make_dataset(
    adapters: List[DatasetAdapter],
    seed = None
) -> Tuple[List[FormattedDatasetRow], List[FormattedDatasetRow]]:
    """
    Make dataset and collator for Direct-Preference Optimization.
    Datasets are expected to have the following columns: {`prompt`, `chosen`, `rejected` }
    Returns a tuple of (train_dataset, eval_dataset)
    """

    train_dataset_list: List[Dataset] = []
    eval_dataset_list: List[Dataset] = []

    # Get all datasets indicated in config.py, then append them to list
    for adapter in adapters:
        assert 'train' in adapter.splits, \
            f"'train' split not found in {adapter.__name__}."

        # TODO: Remove hardcoded 'train' and 'test' splits
        dataset = adapter.load_dataset('train')
        if 'test' in adapter.splits:
            eval_dataset = adapter.load_dataset('test')

            train_dataset_list.append(dataset)
            eval_dataset_list.append(eval_dataset)

        else:
            dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=seed)
            train_dataset_list.append(dataset['train'])
            eval_dataset_list.append(dataset['test'])

    # We concatenate all datasets in list first, then shuffle before getting train eval split
    dataset = concatenate_datasets(train_dataset_list)
    eval_dataset = concatenate_datasets(eval_dataset_list)

    return dataset, eval_dataset
