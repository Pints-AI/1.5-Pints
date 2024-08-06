from pathlib import Path
from torch.utils.data import DataLoader

from lit_gpt.datamodules.slim_orca_idontknow import SlimOrcaIDK
from lit_gpt.utils import CycleIterator
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.datamodules.base import get_sft_collate_fn

tokenizer = Tokenizer(Path('tokenizer/pints_chat'))


def test_slim_orca_idk_e2e():
    batch_size = 2

    slim_orca_idk = SlimOrcaIDK(
        prompt_style='chatml',
        data_files_path='lit_gpt/datamodules/slim-orca-idontknow_test.parquet',
    )
    slim_orca_idk.tokenizer = tokenizer
    slim_orca_idk.include_multiturn_conversations = True

    data = slim_orca_idk.setup()

    train_dataloader = DataLoader(
        data['train_dataset'],
        batch_size=batch_size,
        collate_fn=get_sft_collate_fn(2048),
    )

    slim_orca_idk_iterator = CycleIterator(train_dataloader)

    iter_batch = next(slim_orca_idk_iterator)

    # Check that we plucked out according to batch_size
    assert len(iter_batch['input_ids']) == batch_size
    assert len(iter_batch['labels']) == batch_size

    decoded_input = tokenizer.decode(
        iter_batch['input_ids'][1], skip_special_tokens=False
    )

    expected_input = """<s><|im_start|> user
Chanakya, 4th Century BC Indian political philosopher. The Arthashastra provides an account of the science of politics for a wise ruler, policies for foreign affairs and wars, the system of a spy state and surveillance and economic stability of the state. Chanakya quotes several authorities including Bruhaspati, Ushanas, Prachetasa Manu, Parasara, and Ambi, and described himself as a descendant of a lineage of political philosophers, with his father Chanaka being his immediate predecessor. Another influential extant Indian treatise on political philosophy is the Sukra Neeti. An example of a code of law in ancient India is the Manusmṛti or Laws of Manu.
Where is Sukra Neeti an example of a code of law? (If the question is unanswerable, say "unanswerable")<|im_end|> 
<|im_start|> assistant
Unanswerable. The Sukra Neeti is not an example of a code of law but rather an influential Indian treatise on political philosophy. The example of a code of law in ancient India mentioned is the Manusmṛti or Laws of Manu.<|im_end|>"""

    labels = iter_batch['labels'][1]
    labels[labels == -100] = 0
    decoded_labels = tokenizer.decode(
        iter_batch['labels'][1], skip_special_tokens=False
    )

    expected_labels = """<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>Unanswerable. The Sukra Neeti is not an example of a code of law but rather an influential Indian treatise on political philosophy. The example of a code of law in ancient India mentioned is the Manusmṛti or Laws of Manu.<|im_end|>"""

    assert expected_input == decoded_input
    assert expected_labels == decoded_labels
