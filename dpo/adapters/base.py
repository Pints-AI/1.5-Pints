from multiprocessing import cpu_count
from typing import List, Literal, TypedDict

# `cpu_count() - 2` is to keep 2 cpus for other uses (and 2 because usually 2 virtual cores is one phyiscal core)
# But in non multi-core machines, we just use 1
MAX_USABLE_CORES = min(cpu_count() - 2, 10)  # 10 is the limit
NUM_PROCESSES = max(MAX_USABLE_CORES, 1)

Split = Literal['train', 'test']


class FormattedDatasetRow(TypedDict):
    prompt: str
    chosen: str
    rejected: str


class DatasetAdapter:

    splits: List[Split] = []

    def load_dataset(self, split: Split, **kwargs) -> List[FormattedDatasetRow]:
        """
        Processes and returns the dataset.
        """
        raise NotImplementedError
