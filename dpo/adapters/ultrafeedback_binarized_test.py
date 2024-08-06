from datasets import load_dataset

from dpo.adapters.ultrafeedback_binarized import format_dataset


def test_format_dataset_chatml():
    MOCK_DATA = load_dataset(
        'json',
        split=['train'],
        data_files='./dpo/adapters/ultrafeedback_binarized_test.jsonl',
        streaming=False,
    )[0]

    train_data = format_dataset(
        dataset=MOCK_DATA,
        prompt_template='chatml',
        score_distance=0,
    )

    train_data_row_one = train_data[0]
    mock_data_row_one = MOCK_DATA[0]

    expected_prompt = f"""\
<s><|im_start|>user
{mock_data_row_one['prompt']}<|im_end|>
<|im_start|>assistant
"""

    assert len(MOCK_DATA) == len(train_data)
    expected_chosen = f"""{mock_data_row_one['chosen'][1]["content"]}<|im_end|>"""
    expected_rejected = f"""{mock_data_row_one['rejected'][1]["content"]}<|im_end|>"""

    assert expected_prompt == train_data_row_one['prompt']
    assert expected_chosen == train_data_row_one['chosen']
    assert expected_rejected == train_data_row_one['rejected']
