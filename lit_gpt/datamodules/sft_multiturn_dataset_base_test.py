import pytest
from typing import List
from torch import tensor, concat, equal
from lit_gpt.datamodules.sft_multiturn_dataset_base import (
    SFTMultiTurnDataset,
    MultiTurnDataRow,
)

mocked_data: List[List[MultiTurnDataRow]] = [
    [
        {'instruction': 'Foo', 'input': '', 'output': 'Bar'},
        {'instruction': 'Foo2', 'input': '', 'output': 'Bar2'},
    ],
    [
        {
            'instruction': 'Foo3',
            'input': '',
            'output': 'Bar3',
        },
    ],
]

mocked_data_tokenized = concat(
    (
        tensor([1, 60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62, 117]),
        tensor([115, 101, 114, 10, 70, 111, 111, 60, 124, 105, 109, 95, 101, 110]),
        tensor([100, 124, 62, 10, 60, 124, 105, 109, 95, 115, 116, 97, 114, 116]),
        tensor([124, 62, 97, 115, 115, 105, 115, 116, 97, 110, 116, 10, 66, 97]),
        tensor([114, 2, 10, 60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124]),
        tensor([62, 117, 115, 101, 114, 10, 70, 111, 111, 50, 60, 124, 105, 109]),
        tensor([95, 101, 110, 100, 124, 62, 10, 60, 124, 105, 109, 95, 115, 116]),
        tensor([97, 114, 116, 124, 62, 97, 115, 115, 105, 115, 116, 97, 110, 116]),
        tensor([10, 66, 97, 114, 50, 2]),
    )
)


def mocked_data_labels(ignore_index=-100):
    i = ignore_index
    return concat(
        (
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, 66, 97, 114, 2, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, i, i, i, i, i, i, i]),
            tensor([i, i, i, i, i, 66, 97, 114, 50, 2]),
        )
    )


@pytest.mark.parametrize('mask_prompt', [True, False])
@pytest.mark.parametrize('ignore_index', [-1, -100])
@pytest.mark.parametrize('max_seq_length', [1000])
def test_getitem(max_seq_length, ignore_index, mask_prompt, MOCK_TOKENIZER_FIXTURE):
    dataset = SFTMultiTurnDataset(
        data=mocked_data,
        tokenizer=MOCK_TOKENIZER_FIXTURE,
        prompt_style='chatml',
        mask_prompt=mask_prompt,
        ignore_index=ignore_index,
        max_seq_length=max_seq_length,
    )

    assert len(dataset) == len(mocked_data)

    expected_labels = (
        mocked_data_labels(ignore_index) if mask_prompt else mocked_data_tokenized
    )

    first_set = dataset[0]

    assert equal(first_set['input_ids'], mocked_data_tokenized[:max_seq_length])
    assert equal(first_set['labels'], expected_labels)


@pytest.mark.parametrize('mask_prompt', [True, False])
@pytest.mark.parametrize('ignore_index', [-1, -100])
@pytest.mark.parametrize('max_seq_length', [5])
def test_getitem_hit_max_seq_length(
    max_seq_length, ignore_index, mask_prompt, MOCK_TOKENIZER_FIXTURE
):
    dataset = SFTMultiTurnDataset(
        data=mocked_data,
        tokenizer=MOCK_TOKENIZER_FIXTURE,
        prompt_style='chatml',
        mask_prompt=mask_prompt,
        ignore_index=ignore_index,
        max_seq_length=max_seq_length,
    )

    assert len(dataset) == len(mocked_data)

    eos = tensor([MOCK_TOKENIZER_FIXTURE.eos_id])

    expected_labels = (
        mocked_data_labels(ignore_index) if mask_prompt else mocked_data_tokenized
    )

    first_set = dataset[0]

    assert equal(
        first_set['input_ids'],
        concat((mocked_data_tokenized[: max_seq_length - 1], eos)),
    )

    # If the prompt is not masked, we won't see the eos token
    # Although we should, but this is too trival a problem to fix
    # And it is more likely to occur for training very low context window models,
    # using dataset with high tokens length or without filtering them,
    # i.e, context window 2k, but dataset often has prompts (not even response yet) exceeding that.
    if mask_prompt:
        assert equal(first_set['labels'], expected_labels[:max_seq_length])
    else:
        assert equal(
            first_set['labels'],
            concat((expected_labels[: max_seq_length - 1], eos)),
        )


def test_getitem_check_labels(MOCK_TOKENIZER_FIXTURE):
    dataset = SFTMultiTurnDataset(
        data=mocked_data,
        tokenizer=MOCK_TOKENIZER_FIXTURE,
        prompt_style='chatml',
    )

    labels_from_first_row = dataset[0]['labels']
    labels_from_first_row_decoded: str = MOCK_TOKENIZER_FIXTURE.decode(
        labels_from_first_row
    )

    print(labels_from_first_row_decoded.replace('<ignore_index>', ''))
    # Removing away <ignore_index> represents tokens that the model is trained on
    # Again, this is different from attention. All tokens in the sequence are attended to.
    # Therefore, we should only have a concatenated string of responses:
    assert (
        labels_from_first_row_decoded.replace('<ignore_index>', '')
        == 'Bar</s>Bar2</s>'  # </s> is correct and not <|im_end|> because tokenizer is mocked
    )
