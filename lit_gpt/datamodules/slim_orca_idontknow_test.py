from lit_gpt.datamodules.slim_orca_idontknow import format_dataset, SlimOrcaIDK

slim_orca_idk = SlimOrcaIDK(
    data_files_path='lit_gpt/datamodules/slim-orca-idontknow_test.parquet'
)
slim_orca_idk_dataset = slim_orca_idk.prepare_data()
MOCK_DATA = slim_orca_idk_dataset[0]


def test_slim_orca_idontknow_format_dataset():
    train_data = format_dataset(
        dataset=MOCK_DATA, include_multi_turn_conversations=None
    )

    train_data_row_one = train_data[0]
    mock_data_row_one = MOCK_DATA[0]['conversations']

    # The instruction and output pair in train_data is made
    # from `content` from the nth and nth+1 row in `messages``
    assert mock_data_row_one[1]['value'] == train_data_row_one['instruction']
    assert mock_data_row_one[2]['value'] == train_data_row_one['output']
