from pathlib import Path
from typing import Optional, List, TypedDict
from glob import glob
from pandas import read_parquet
from .preparer import DatasetPreparer

# Name of dataset
DATASET_NAME = 'pints-expository-prose-v1'


class StandardParquet(TypedDict):
    source: List[str]
    source_id: List[str]
    text: List[str]


class StandardParquetPreparer(DatasetPreparer):
    def collect_files(self, full_source_path: Path):
        return glob(f'{full_source_path}/**/*.parquet', recursive=True)

    def read_file(
        self, filepath: Path, parquet_columns: Optional[List[str]] = None
    ) -> StandardParquet:
        contents = read_parquet(filepath, engine='pyarrow', columns=parquet_columns)
        return contents

    def read_file_contents(self, filepath: Path) -> List[str]:
        data = self.read_file(filepath, ['text'])
        return data['text']


prepare_dataset = StandardParquetPreparer(DATASET_NAME)


def main(
    source_path: Optional[Path] = None,
    tokenizer_path=prepare_dataset.tokenizer_path,
    destination_path=prepare_dataset.destination_path,
    chunk_size=prepare_dataset.chunk_size,
    percentage=prepare_dataset.percentage,
    train_val_split_ratio=prepare_dataset.train_val_split_ratio,
    max_cores: Optional[int] = None,
) -> None:
    prepare_dataset.prepare(
        source_path,
        tokenizer_path,
        destination_path,
        chunk_size,
        percentage,
        train_val_split_ratio,
        max_cores,
    )


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(main)
