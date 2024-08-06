from typing import Callable, List, Optional, Union, TypedDict
import torch
from torch import Tensor
from lit_gpt.datamodules.sft_dataset_base import SFTDataset
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.prompts import PromptStyle


class MultiTurnDataRow(TypedDict):
    instruction: str
    input: str
    output: str


class MultiTurnDataItem(TypedDict):
    input_ids: Tensor
    labels: Tensor


class SFTMultiTurnDataset(SFTDataset):
    """A multiturn version of `SFTDataset`, which is an in-memory dataset for supervised finetuning with `input_ids` and `labels`.

    Args:
        data: A list of samples (dicts). The target/label must be stored under the key 'output' and the instruction
            or other data can be stored under any key as long as it is compatible with the given prompt template.
        tokenizer: The tokenizer to use. Should match the one that was used to pretrain the model.

        The data structure looks like this:
        [
          [
            {'instruction': '1st turn, user says something.', 'input': '...', 'output': '2nd turn, assistant responds.'},
            {'instruction': '3rd turn, user replies.', 'input': '...', 'output': '4th turn, assistant responds again.'},
            ... (nth more turns)
          ],
          [ ...The second set... ]
        ]

        prompt_style: The style to apply to prompts. See `litgpt.prompts` for a list of available styles.
        max_seq_length: Truncate sequences that are longer than this value. By default, no truncation is applied.
        mask_prompt: Whether to mask the prompt section from the label (with ``ignore_index``).
        ignore_index: The index to use for elements to be ignored in the label.
        transform: An optional transform to apply to the sample before it gets tokenized. Use this to rename the
            keys in the dataset to the expected 'instruction' and 'output' keys.

    Returns a dict with two keys:
        input_ids: The encoded prompt + response
        labels: Same as input_ids, unless ``mask_prompt=True`` in which case the 'prompt' part is replaced with
            the ``ignore_index``.
    """

    # Overwrite the base class' data type with our custom MultiTurn type.
    # Read docstring above to visualise the data structure.
    data: List[List[MultiTurnDataRow]]

    # More strictly type the transform function to overwrite the base class, and avoid `Any`
    transform: Optional[Callable[[List[MultiTurnDataRow]], List[MultiTurnDataRow]]]

    def __init__(
        self,
        data: List[List[MultiTurnDataRow]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[
            Callable[[List[MultiTurnDataRow]], List[MultiTurnDataRow]]
        ] = None,
    ) -> None:
        super().__init__(
            data,
            tokenizer,
            prompt_style,
            max_seq_length,
            mask_prompt,
            ignore_index,
            transform,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> MultiTurnDataItem:
        example = self.data[idx]

        if self.transform is not None:
            example = self.transform(example)

        all_input_ids = None
        all_labels = None

        for i, turn_pair in enumerate(example):
            # To allow application of IGNORE_INDEX to ignore CE losses for prompts,
            # we construct both the prompt with output, and without output.
            # Then, we can "diff" the tokens later, and apply IGNORE_INDEX to the encoded prompts,
            # to arrive at our `labels`.
            #
            # WHY?
            #
            # We want to teach the model to learn from responses only, and avoid learning
            # the styles of the prompt, which can be deliberately low in quality.
            # If we don't ignore the CE loss for the prompts, the model learn to speak like the prompt,
            # and that's not desired, like broken English, or CSV files from a RAG dataset.
            #
            # Broken English example:-
            # User prompt: can u tell me y so sadddd...
            # AI response/output: I'm sorry that you are sad, and certainly hope you...
            #
            # RAG Example:-
            # User prompt: file.csv: timedate,name,address,amount,2021-01-01:0800Z,john,doe...
            # AI response/output: I see that you have provide
            prompt = self.prompt_style.apply(
                prompt=turn_pair['instruction'], **turn_pair
            )

            if i > 0:
                prompt = '\n' + prompt

            prompt_and_response = prompt + turn_pair['output']

            add_bos = True if i == 0 else False

            encoded_prompt = self.tokenizer.encode(
                prompt, bos=add_bos, max_length=self.max_seq_length
            )

            encoded_prompt_and_response = self.tokenizer.encode(
                prompt_and_response,
                bos=add_bos,
                eos=True,
                max_length=self.max_seq_length,
            )

            # The labels are the full prompt with response, but with the prompt masked out
            labels = encoded_prompt_and_response.clone()

            # The term `mask` is confusing here.
            # The prompt is not masked, it's just ignored (skipped over) for Cross-entropy (CE) calculation
            # So attention scores are still computed, hence technically not masked.
            if self.mask_prompt:
                labels[: len(encoded_prompt)] = self.ignore_index

            # If the turn pair is too long, the tokenizer would truncate
            # and the data ends halfway through the user's prompt.
            if len(encoded_prompt) == len(encoded_prompt_and_response):
                # print out and warn user
                message = (
                    'Dataset is truncated. The output will not be seen by the model.'
                )
                message += ' If you receive too many of these messages, consider preprocessing the dataset to reduce sequence length.'
                print(message)

            # Drop subsequent turns if max sequence length is reached.
            if (
                self.max_seq_length > 0  # Max sequence is implemented
                and i
                > 0  # Do for 2nd turn pairs onwards. The first pairs would have been truncated by the tokenizer
                and len(all_input_ids)
                + len(
                    encoded_prompt_and_response
                )  # If adding this pair will exceed max sequence.
                > self.max_seq_length
            ):
                break

            # Append to the inputs and labels.
            if i == 0:
                all_input_ids = encoded_prompt_and_response.type(torch.int64)
                all_labels = labels.type(torch.int64)
            else:
                all_input_ids = torch.cat(
                    (all_input_ids, encoded_prompt_and_response.type(torch.int64))
                )
                all_labels = torch.cat((all_labels, labels.type(torch.int64)))

        if all_input_ids is None:
            message = 'WARN: SFTMultiTurnDataset hit and empty row! '
            message += ' Trying to return empty data so that training can continue.'
            message += ' If you see too many of these, your training will be adversely affected.'
            message += ' Check your dataset, or the adapters.'
            print(message)

            return {
                'input_ids': torch.tensor([self.tokenizer.eos_id], dtype=torch.int64),
                'labels': torch.tensor([self.ignore_index], dtype=torch.int64),
            }

        return {
            'input_ids': all_input_ids,
            'labels': all_labels,
        }
