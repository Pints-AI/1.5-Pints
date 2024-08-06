# conftest.py
import pytest
from unittest.mock import MagicMock
import sys
import torch
from typing import Optional

# xformers.ops
mock_swiGLU = MagicMock(name='SwiGLU')
xformers_ops_mock = MagicMock(SwiGLU=mock_swiGLU)
sys.modules['xformers.ops'] = xformers_ops_mock

# FLASH ATTENTION
# Without CUDA, all these needs to be mocked.

# lit_gpt/fused_rotary_embedding
sys.modules['rotary_emb'] = MagicMock()
sys.modules['einops'] = MagicMock()

# lit_gpt/fused_cross_entropy
sys.modules['xentropy_cuda_lib'] = MagicMock()


class MockTokenizer:
    """A dummy tokenizer that encodes each character as its ASCII code."""

    bos_id = 1
    eos_id = 2

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = False,
        max_length=-1,
    ) -> torch.Tensor:
        tokens = [ord(c) for c in string]
        if bos:
            bos_id = self.bos_id
            if bos_id is None:
                raise NotImplementedError(
                    'Attempted to add bos, but this tokenizer does not defined a bos token.'
                )
            tokens = [bos_id] + tokens

        if max_length > 0:
            if eos:
                if self.eos_id is None:
                    raise NotImplementedError(
                        'Attempted to add eos, but this tokenizer does not defined an eos token'
                    )
                if len(tokens) >= max_length:
                    tokens = tokens[:max_length]
                    tokens[-1] = self.eos_id
                else:
                    tokens = tokens + [self.eos_id]
        else:
            if eos:
                if self.eos_id is None:
                    raise NotImplementedError(
                        'Attempted to add eos, but this tokenizer does not defined an eos token'
                    )
                tokens = tokens + [self.eos_id]

        return torch.tensor(tokens)

    def decode(self, tokens: torch.Tensor) -> str:
        decoded = ''
        for token in tokens.tolist():
            if token == -100:
                decoded += '<ignore_index>'
                continue

            if token == 1:
                decoded += '<s>'
                continue

            if token == 2:
                decoded += '</s>'
                continue

            decoded += chr(int(token))

        return decoded

        # return ''.join(chr(int(t)) for t in tokens.tolist())


@pytest.fixture()
def MOCK_TOKENIZER_FIXTURE():
    return MockTokenizer()
