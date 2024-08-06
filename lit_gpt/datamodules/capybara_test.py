from json import loads
from lit_gpt.datamodules.capybara import format_dataset

MOCK_DATA = []

with open(
    './lit_gpt/datamodules/capybara_test.jsonl', 'r', encoding='utf-8'
) as jsonl_file:
    for line in jsonl_file:
        MOCK_DATA.append(loads(line))


def test_capybara_format_dataset_multiturn():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=True
    )

    train_data_row_one = train_data[0]
    mock_data_row_one = MOCK_DATA[0]['conversation']

    # Multiturn row check
    # Each data row has an input/output pair, hence 1 to 1 conversion
    assert len(mock_data_row_one) == len(train_data_row_one)

    # The instruction and output pair in train_data is made
    # from `content` from the nth and nth+1 row in `messages``
    assert mock_data_row_one[0]['input'] == train_data_row_one[0]['instruction']
    assert mock_data_row_one[0]['output'] == train_data_row_one[0]['output']
    assert len(train_data_row_one) == len(mock_data_row_one)


def test_capybara_format_dataset():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=False
    )

    train_data_row_two = train_data[1]
    mock_data_row_two = MOCK_DATA[1]['conversation']

    assert mock_data_row_two[0]['input'] == train_data_row_two[0]['instruction']
    assert mock_data_row_two[0]['output'] == train_data_row_two[0]['output']

    # Because we don't include multiturn,
    # the behaviour is that only 1 instruction/output pair is made,
    assert len(train_data_row_two) == 1
    # despite the dataset having more than 1 pair
    assert len(mock_data_row_two) > 1
