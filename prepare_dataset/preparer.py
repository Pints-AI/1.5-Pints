from multiprocessing import Process, cpu_count, Queue
from pathlib import Path
from typing import Optional, List, Union
import time
import os
import logging
from tqdm import tqdm
import numpy
import pandas
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.packed_dataset import PackedDatasetBuilder
from util.splitter import Splitter
import time


# =========
# DEFAULTS
# =========
# Initialize only the ones with defaults
TOKENIZER_PATH = Path('tokenizer/pints')

# `CHUNK_SIZE` refers to the number of tokens to pack into 1 file.
# This should be the (context window + 1) * n, where n is just to define how much to pack into a file.
# For explanation about "n", see https://github.com/Lightning-AI/litgpt/issues/907
# Also see https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset
CONTEXT_LENGTH = 16384
SEQUENCES_PER_FILE = 1024
# We need context window + 1 because for the last token of the context window, it needs to predict the next token.
CHUNK_SIZE = (CONTEXT_LENGTH + 1) * SEQUENCES_PER_FILE


# The amount to take from the dataset. 0.5 means 50%.
PERCENTAGE = 1.0

# 0.8 means 80% will be retained as training data, and 20% as validation
TRAIN_VAL_SPLIT_RATIO = 1.0


class ErrorCountHandler(logging.Handler):
    """
    This is to be attached to our logging interface.
    Useful at the end of dataset preparation, to tell user how many errors are there.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.error_count = 0

    def emit(self, record):
        if record.levelno == logging.ERROR:
            self.error_count += 1


class DatasetPreparer:
    # Create a queue as a mean to consolidate progress for tqdm bar
    progress_queue = Queue()
    Splitter = Splitter
    tokenizer_path = TOKENIZER_PATH
    chunk_size = CHUNK_SIZE
    percentage = PERCENTAGE
    train_val_split_ratio = TRAIN_VAL_SPLIT_RATIO

    # Sometimes preparation can OOM due to too many process
    # Use this to limit the cores
    max_cores: Union[int, None] = None

    error_count_handler = ErrorCountHandler()

    def __init__(self, dataset_name: str):
        """
        DatasetPreparer is a pluggable/hackable class to abstract data handling strategies.

        Example:

        class MyOwnPreparer(DatasetPreparer):
            def collect_files(sef, full_source_path: Path):
                # define how to collect the files from the dataset

        preparer = MyOwnPreparer('source/path')

        # swop out your own plugins
        preparer.Splitter = MyOwnSplitter
        """

        # String safe dataset_name
        assert '/' not in dataset_name, 'Slash ("/") not allowed in data_name.'
        assert '\\' not in dataset_name, 'Slash ("\\") not allowed in data_name.'

        self.dataset_name = dataset_name
        self.destination_path = Path('data/output') / dataset_name

        # Isolated logging instance to output to error log file
        logger = logging.getLogger(dataset_name)
        logger.setLevel(logging.ERROR)

        # Set up logs
        logpath = f'logs/{dataset_name}.error.log'
        Path('logs').mkdir(parents=True, exist_ok=True)
        logfile_handler = logging.FileHandler(logpath)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
        )
        logfile_handler.setFormatter(formatter)
        logger.addHandler(logfile_handler)

        # Set up counter
        logger.addHandler(self.error_count_handler)

        self.logger = logger
        self.logger_logpath = logpath

    def prepare(
        self,
        source_path: Optional[Path] = None,
        tokenizer_path=TOKENIZER_PATH,
        destination_path: Optional[Path] = None,
        chunk_size=CHUNK_SIZE,
        percentage=PERCENTAGE,
        train_val_split_ratio=TRAIN_VAL_SPLIT_RATIO,
        max_cores=None,
    ) -> None:
        """
        Prepare dataset

        Args:
            source_path: The path where your dataset is.
            train_eval_split: Split dataset into train and eval. 1.0 means no dataset will be split for validation
            max_cores: Sometimes preparation can OOM due to too many process, use this to limit the cores.
        """
        assert source_path is not None, 'Please provide source_path.'
        assert percentage > 0, "Percentage of 0 and less of dataset don't make sense."
        assert percentage <= 1.0, 'Percentage cannot be more than 1.0.'
        assert (
            train_val_split_ratio >= 0.5
        ), "Spiltting dataset where training data is less than half doesn't make sense."
        assert train_val_split_ratio <= 1.0, 'Split ratio cannot be more than 1.0.'
        assert source_path is not None, 'Please provide source_path.'

        if destination_path is not None:
            self.destination_path = destination_path
        else:
            self.destination_path = destination_path = (
                Path('data/output') / self.dataset_name
            )
            print(f'No destination path provided. Defaulting to {destination_path}')

        self.tokenizer_path = tokenizer_path
        self.chunk_size = chunk_size
        self.percentage = percentage
        self.train_val_split_ratio = train_val_split_ratio
        self.max_cores = max_cores

        # Print and use full path so that it's easy to debug, and no ambiguity
        full_source_path = os.path.normpath(os.path.join(os.getcwd(), source_path))
        full_source_path = Path(full_source_path)

        assert (
            full_source_path.exists()
        ), f'`source_path` [{full_source_path}] does not exist.'

        filepaths = self.collect_files(full_source_path)
        filepaths = sorted(filepaths)
        no_of_files = len(filepaths)

        assert no_of_files > 0, f'No files found in `source_path` [{full_source_path}].'

        # Data operations is mandane and error prone. Make user double check.
        print('====FILES FOUND====')
        print(filepaths[:20])
        if no_of_files > 20:
            print(f'     (and [{no_of_files - 20}] more files...)')
        print('===================')
        print('PLEASE CHECK IF DATASOURCE IS CORRECT:')
        print(f'Taking data from [{full_source_path}]')
        print(f'Total of [{no_of_files}] files in there.')

        is_ok = input('(yes/no)\n')

        if is_ok != 'yes':
            print("You did not answer 'yes', exiting...")
            return

        # After joining with `os.path.join`, the .. will still be in path.
        # `os.path.normpath` will remove that.
        full_output_path = os.path.normpath(os.path.join(os.getcwd(), destination_path))
        full_output_path = Path(full_output_path)
        print(f'Output files will be stored in [{full_output_path}]')

        if destination_path.is_dir() and len(os.listdir(destination_path)) > 0:
            is_ok = input(
                'DANGER! It is not empty. Files may be overwritten. Continue? (yes/no)\n'
            )

            if is_ok != 'yes':
                print("You did not answer 'yes', exiting...")
                return

        self.check_data(filepaths[0])
        
        self._check_tokenizer_for_pad(tokenizer_path)

        if percentage < 1:
            filepaths = self.subsample(filepaths)

        num_processes = self._get_num_processes(no_of_files)

        start_time = time.time()

        self._distribute(filepaths, num_processes)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Time taken: {elapsed_time:.2f} seconds')

        error_count = self.error_count_handler.error_count
        if error_count > 0:
            print(
                f'Encountered [{error_count}] error{"s" if error_count > 1 else ""}. Check logs at [{self.logger_logpath}]'
            )

    def collect_files(self, full_source_path: Path) -> List[str]:
        """This defines how the file will be collected."""

        raise NotImplementedError(
            'You need to implement how to read and access the dataset.'
        )

    def check_data(self, filepath: Path):
        """Check the data once."""

        contents = self.read_file_contents(filepath)
        print('\n\nPrinting out an example. Please do a check.')
        print(f'Source: [{filepath}]')
        print('===================================================')
        selected_content = contents[0]
        peek = 1000
        if len(selected_content) > peek:
            print(selected_content[:peek])
            print(f'\n\n  (...and {len(selected_content) - 1000} more characters.)')
        else:
            print(contents[0])
        print('===================================================')
        is_ok = input('Does the data look correct? (yes/no)\n')
        if is_ok != 'yes':
            print("You did not answer 'yes'. Exiting..")
            exit()

    def read_file(
        self,
        filepath: Path,
        # Parquet allows sub selection by columns
        # Using this will boost a lot of IO and mem efficiencies.
        parquet_columns: Optional[List[str]] = None,
    ):
        """
        This defines how the file will be accessed.
        NOTE: This does not mandate a return type, as different files will return different types.
        """

        raise NotImplementedError(
            'You need to implement how to read and access the dataset.'
        )

    def read_file_contents(self, filepath: Path) -> List[str]:
        """
        This defines how to get the main contents.
        Is an abstraction over read_file, which is used to access anything in the file.
        NOTE: This function mandates that return type be List[str].
        """
        # data = self.read_file(filepath)

        raise NotImplementedError(
            'You need to implement how to get content from the dataset.'
        )

    def subsample(self, filepaths: List[Path]) -> List[Path]:
        """
        Simple subsampling using pandas.
        This is deterministic, i.e. For same input, the selected files will always be the same.
        """

        print(f'Subsampling... started with [{len(filepaths)}] files.')
        # Using pandas to subsample. Using `random_state=888` makes subsampling reproducible
        dataframe = pandas.DataFrame(filepaths, columns=['filepaths'])

        # the subsampling rate should be compensated with the split ratio
        print(f'Subsampling: [{self.percentage}]')
        subsampling_rate = self.percentage / self.train_val_split_ratio
        print(f'Subsampling (train/val compensated): [{subsampling_rate}]')
        if subsampling_rate > 1.0:
            print(
                'Subsampling rate is more than 1.0. There is not enough to compensate.'
            )
            print('No subsampling will happen. Consider lowering train/val split.')
            return filepaths

        subsampled = dataframe.sample(frac=subsampling_rate, random_state=888)
        filepaths = subsampled['filepaths'].tolist()
        print(f'After subsampling: [{len(filepaths)}] files')
        return filepaths

    def _distribute(
        self,
        filepaths: List[Path],
        num_processes: int,
    ):
        """Distribute the files amongst available processes."""

        print(f'Spilting [{len(filepaths)}] files to [{num_processes}] processes...')
        print('This might take awhile...')
        # Now chunk the filespaths to distribute to the processes
        chunked_filepaths = numpy.array_split(filepaths, num_processes)
        print(f'Split done. Got [{len(chunked_filepaths)}] chunks of:')
        for i, chunk in enumerate(chunked_filepaths):
            print(f'Chunk[{i}]: [{len(chunk)}]')

        processes: List[Process] = []

        # Distribute the tasks
        for process_id, subset in enumerate(chunked_filepaths):
            print(f'Distributing to process[{process_id}]...')

            # convert NDArray back to a list, easier.
            subset = list(subset)
            process = Process(
                target=self.process_data,
                args=(
                    self.tokenizer_path,
                    self.destination_path,
                    self.chunk_size,
                    self.train_val_split_ratio,
                    subset,
                    self.progress_queue,
                    process_id,
                ),
            )
            process.start()
            print(f'Distributed to process[{process_id}].')
            processes.append(process)

        print('Successfully distributed.')

        # Wait for all workers to finish
        for process in processes:
            process.join()

    def process_data(
        self,
        tokenizer_path: Path,
        destination_path: Path,
        chunk_size: int,
        train_val_split_ratio: float,
        filepaths: List[str],
        progress_queue: Queue,
        process_id: int = 0,
    ) -> None:
        previous_iteration_time = None
        total_tasks = len(filepaths)
        tasks_completed = 0

        try:
            assert len(filepaths) > 0, 'No files provided.'

            training_outdir = destination_path / 'train'

            # If path don't exist make it.
            training_outdir.mkdir(parents=True, exist_ok=True)

            tokenizer = Tokenizer(tokenizer_path)
            
            training_dataset_builder = PackedDatasetBuilder(
                outdir=training_outdir,
                # Use process_id to differentiate builders
                prefix=f'train_{self.dataset_name}_{process_id}',
                chunk_size=chunk_size,
                # NOTE: `sep_token` does not work as it says.
                # See https://github.com/Lightning-AI/lit-llama/issues/482
                # Consequently, it a token that fills up the initial tensor.
                # And works more like a pad token.
                # Also see issue: https://github.com/jzhang38/TinyLlama/issues/83
                pad_token=tokenizer.pad_id,
                dtype='auto',
                vocab_size=tokenizer.vocab_size,
            )

            validation_dataset_builder = None
            splitter = None

            if train_val_split_ratio < 1.0:
                validation_outdir = destination_path / 'validation'
                validation_outdir.mkdir(parents=True, exist_ok=True)
                validation_dataset_builder = PackedDatasetBuilder(
                    outdir=validation_outdir,
                    # Use process_id to differentiate builders
                    prefix=f'validation_{self.dataset_name}_{process_id}',
                    chunk_size=chunk_size,
                    pad_token=tokenizer.pad_id,
                    dtype='auto',
                    vocab_size=tokenizer.vocab_size,
                )
                splitter = Splitter(train_val_split_ratio)

            for filepath in filepaths:
                try:
                    self._build_data(
                        filepath,
                        tokenizer,
                        training_dataset_builder,
                        validation_dataset_builder,
                        splitter,
                    )

                except Exception as error:
                    error_msg = f'Process[{process_id}] Error with buidling data for file[{filepath}]'
                    self.logger.error(error_msg)
                    self.logger.error(error)

                finally:
                    tasks_completed += 1
                    time_taken = None
                    current_time = time.time()

                    if previous_iteration_time is None:
                        previous_iteration_time = current_time
                    else:
                        time_taken = current_time - previous_iteration_time

                    formatted_time = time.localtime(current_time)
                    formatted_time = time.strftime('%H:%M:%S', formatted_time)

                    message = f'{formatted_time}'
                    message += f' Process id[{process_id}]:'
                    message += f' Progress: {tasks_completed}/{total_tasks}'
                    if total_tasks == tasks_completed:
                        message += ' [DONE!]'
                    if time_taken:
                        message += f' Time taken: {time_taken:.2f}s'

                    print(message)

                    previous_iteration_time = current_time

        except Exception as error:
            error_msg = f'Process ID[{process_id}] failed completely with error:'
            self.logger.error(error_msg)
            self.logger.error(error)
            raise error

    def _build_data(
        self,
        filepath: Path,
        tokenizer: Tokenizer,
        training_dataset_builder: PackedDatasetBuilder,
        validation_dataset_builder: Union[PackedDatasetBuilder, None],
        splitter: Splitter,
    ):
        contents = self.read_file_contents(filepath)

        # Encode away~
        for row_number, content in enumerate(contents):
            # uncomment this to check the contents
            # print(content)

            try:
                # Previously tokenizer.encode defaults to eos=True, now it defaults to False, due to downstream issues.
                # Given how messy this is, we should always be explicit.
                tokens = tokenizer.encode(content, bos=True, eos=True)
                dataset = numpy.array(tokens, dtype=training_dataset_builder.dtype)

                if splitter is None:
                    training_dataset_builder.add_array(dataset)
                    continue

                if splitter.should_split():
                    validation_dataset_builder.add_array(dataset)
                else:
                    training_dataset_builder.add_array(dataset)

            except Exception as error:
                error_msg = (
                    f'Error building for file[{filepath}], row[{row_number + 1}].'
                )
                error_msg += f'\nFirst 50 characters of content:\n{content[:50]}'
                self.logger.error(error_msg)
                self.logger.error(error)
                continue

        # The builder's #add_array method will write the chunks to files whenever it fills up
        # to the max length.
        # After the loop is complete, there will be some remaining.
        # So we do this to push the final part in.
        training_dataset_builder.write_remainder()
        validation_dataset_builder.write_remainder()

    def _get_num_processes(self, no_of_files: int):
        num_processes = cpu_count() - 1  # Good habit to leave 1 core.

        # Single core machine?? That's sad.
        if num_processes < 1:
            num_processes = 1

        # Limit by max_cores
        if self.max_cores is not None and num_processes > self.max_cores:
            num_processes = self.max_cores

        print(f'No. of cpus to use: {num_processes}')

        if no_of_files < num_processes:
            num_processes = no_of_files
            print(
                f'There are less files than there are cpus, hence using {no_of_files} cpus.'
            )

        print(f'No. of parallel processes: {num_processes}')
        return num_processes

    def _check_tokenizer_for_pad(self, tokenizer_path: Path):
        tokenizer = Tokenizer(tokenizer_path)

        # We need pad_token to work.
        print('Checking if pad_token is valid. We need it for pretraining.')
        pad_token_id = tokenizer.pad_id
        pad_token = None

        # Now check pad token.
        if not isinstance(pad_token_id, int):
            raise Exception(
                f'Invalid `pad_token_id` of "{pad_token_id}" found. Please edit tokenizer.json to add it.'
            )

        # Sometimes it has an index, but cannot be decoded.
        try:
            pad_token = tokenizer.decode([tokenizer.pad_id], False)
            print('Pad token is: ', pad_token)
        except Exception:
            raise Exception('Invalid `pad_token`, it cannot be decoded.')
