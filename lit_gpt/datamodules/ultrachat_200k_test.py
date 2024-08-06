from json import loads
from lit_gpt.datamodules.ultrachat_200k import format_dataset

MOCK_DATA = []

with open(
    './lit_gpt/datamodules/ultrachat_200k_test.jsonl', 'r', encoding='utf-8'
) as jsonl_file:
    for line in jsonl_file:
        MOCK_DATA.append(loads(line))


def test_format_dataset_multiturn():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=True
    )

    train_data_row_one = train_data[0]
    mock_data_row_one = MOCK_DATA[0]['messages']

    # Multiturn row check
    # It comes in a flat list of user-assistant pair, that is turned into 1 train data row
    # Hence divde by 2
    assert len(mock_data_row_one) / 2 == len(train_data_row_one)

    # The instruction and output pair in train_data is made
    # from `content` from the nth and nth+1 row in `messages``
    assert train_data_row_one[0]['instruction'] == mock_data_row_one[0]['content']
    assert train_data_row_one[0]['output'] == mock_data_row_one[1]['content']

    # Because 2 rows of `content` from the data is condensed into 1 row of train_data:
    assert len(train_data_row_one) == len(mock_data_row_one) / 2


def test_format_dataset():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=False
    )

    train_data_row_two = train_data[1]
    mock_data_row_two = MOCK_DATA[1]['messages']

    assert train_data_row_two[0]['instruction'] == mock_data_row_two[0]['content']
    assert train_data_row_two[0]['output'] == mock_data_row_two[1]['content']

    # Because we don't include multiturn,
    # the behaviour is that only 1 instruction/output pair is made,
    assert len(train_data_row_two) == 1
    # despite the dataset having more than 1 pair
    assert len(mock_data_row_two) > 2
