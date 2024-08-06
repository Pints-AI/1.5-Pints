import os
from pathlib import Path
from unittest import TestCase

import numpy as np

from lit_gpt.packed_dataset import PackedDatasetBuilder
from lit_gpt.tokenizer import Tokenizer


class TestTokenizer(TestCase):
    def setUp(self) -> None:
        self.test_string = 'Hello, world!'
        self.tokenizer = Tokenizer(Path('../tokenizer/pints'))

    def test_single_bos_eos(self):
        """
        Test that the tokenizer correctly encodes a single string with bos and eos tokens.
        """
        encoded = self.tokenizer.encode(self.test_string, eos=True, bos=True).tolist()
        # bos + encoded string + eos
        correct_single = (
            [self.tokenizer.bos_id]
            + self.tokenizer.encode(self.test_string, eos=False, bos=False).tolist()
            + [self.tokenizer.eos_id]
        )
        self.assertEqual(encoded, correct_single)

    def test_packed_with_bos_eos(self):
        """
        Test that the packed dataset is correctly formatted with bos and eos tokens, even when multiple sequences are
        packed into it.
        """

        chunk_size = 40
        pad_token = 0
        OUT_DIR = '../tokenizer/pints_chat/FOO'

        os.makedirs(OUT_DIR, exist_ok=True)

        training_dataset_builder = PackedDatasetBuilder(
            outdir=OUT_DIR,
            # Use process_id to differentiate builders
            prefix='BAR',
            chunk_size=chunk_size,
            pad_token=pad_token,  # need to added pad tokens to llama
            dtype='auto',
            vocab_size=self.tokenizer.vocab_size,
        )

        single_encoded = self.tokenizer.encode(
            self.test_string, eos=True, bos=True
        ).tolist()

        # Pack the string into dataset
        training_dataset_builder.add_array(np.array(single_encoded))

        # Get list representation of packed dataset
        packed_single = training_dataset_builder._arr.tolist()

        # Get the correct packed representation
        correct_single = single_encoded + [pad_token] * (
            chunk_size - len(single_encoded)
        )

        # Check that the packed dataset is correct
        assert packed_single == correct_single

        # Pack the string into dataset again
        training_dataset_builder.add_array(np.array(single_encoded))
        test_packed_double = training_dataset_builder._arr.tolist()
        single_encoded.extend(single_encoded)
        correct_packed_double = single_encoded + [
            pad_token for i in range(chunk_size - len(single_encoded))
        ]

        print(self.tokenizer.decode(training_dataset_builder._arr))

        assert test_packed_double == correct_packed_double
