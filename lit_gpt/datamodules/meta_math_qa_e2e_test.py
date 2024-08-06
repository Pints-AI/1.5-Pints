from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

from lit_gpt.datamodules.meta_math_qa import MetaMathQA
from lit_gpt.utils import CycleIterator
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.datamodules.base import get_sft_collate_fn

tokenizer = Tokenizer(Path('tokenizer/pints'))


def test_metamath_e2e():
    batch_size = 2

    metamath = MetaMathQA(prompt_style='chatml')
    metamath.tokenizer = tokenizer
    metamath.include_multiturn_conversations = True

    # overwrite the prepare_data method to load custom
    def mock_prepare_data():
        # We do it like this to make it return a (train, test) tuple that Deita has.
        mock_train = load_dataset(
            'json',
            # Even though jsonl file doesn't have a split, `train` is needed by default
            split=['train'],
            data_files='lit_gpt/datamodules/meta_math_qa_test.jsonl',
        )

        mock_test = load_dataset(
            'json',
            # Even though jsonl file doesn't have a split, `train` is needed by default
            split=['train'],
            data_files='lit_gpt/datamodules/meta_math_qa_test.jsonl',
        )
        return (mock_train[0], mock_test[0])

    metamath.prepare_data = mock_prepare_data

    data = metamath.setup()

    train_dataloader = DataLoader(
        data['train_dataset'],
        batch_size=batch_size,
        collate_fn=get_sft_collate_fn(2048),
    )

    deita_iterator = CycleIterator(train_dataloader)

    iter_batch = next(deita_iterator)

    # Check that we plucked out according to batch_size
    assert len(iter_batch['input_ids']) == batch_size
    assert len(iter_batch['labels']) == batch_size

    inputs_ids = iter_batch['input_ids'][1]
    inputs_ids[inputs_ids == -100] = 0

    decoded_input = tokenizer.decode(inputs_ids, skip_special_tokens=False)

    expected_input = """<s><|im_start|> user
What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?<|im_end|> 
<|im_start|> assistant
Each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80.
So the total cost for each player is $25 + $15.20 + $6.80 = $47.
Since there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752.
#### 752
The answer is: 752<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"""

    labels = iter_batch['labels'][1]
    labels[labels == -100] = 0
    decoded_labels = tokenizer.decode(
        iter_batch['labels'][1], skip_special_tokens=False
    )

    expected_labels = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80.
So the total cost for each player is $25 + $15.20 + $6.80 = $47.
Since there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752.
#### 752
The answer is: 752<|im_end|><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>"""

    print('=' * 80)
    print(decoded_input)
    print('=' * 80)
    print(decoded_labels)
    print('=' * 80)

    assert expected_input == decoded_input
    assert expected_labels == decoded_labels
