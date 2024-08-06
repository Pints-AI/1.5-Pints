import json
from pathlib import Path
from typing import Optional, Union
import torch
from os.path import normpath, join
from os import getcwd

HUGGINGFACE = 'huggingface'
SENTENCEPIECE = 'sentencepiece'


class Tokenizer:
    def __init__(self, checkpoint_dir: Path) -> None:
        # some checkpoints have both files, `.model` takes precedence
        # TODO: Deprecate SentencePieceProcessor. It behaves differently from tokenizer.json
        # So it's best to avoid and reduce complexity.
        # For example, see: https://github.com/google/sentencepiece/issues/667
        # if (vocabulary_path := checkpoint_dir / 'tokenizer.model').is_file():
        #     print(
        #         f'Tokenizer class is initialised with `{vocabulary_path}` as this takes precedence.'
        #     )
        #     print(
        #         'If you have intended to use `tokenizer.json`, please remove `tokenizer.model`.'
        #     )
        #     from sentencepiece import SentencePieceProcessor

        #     self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
        #     self.backend = SENTENCEPIECE
        #     self.bos_id = self.processor.bos_id()
        #     self.eos_id = self.processor.eos_id()
        #     self.pad_id = self.processor.pad_id()

        #     self.processor.Decode

        if (vocabulary_path := checkpoint_dir / 'tokenizer.json').is_file():
            print(f'Tokenizer class is initalised with `{vocabulary_path}`.')
            from tokenizers import Tokenizer as HFTokenizer

            self.processor: HFTokenizer = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = HUGGINGFACE

            with open(checkpoint_dir / 'tokenizer_config.json') as fp:
                config = json.load(fp)

            bos_token_config = config.get('bos_token', None)
            bos_token = (
                bos_token_config
                # bos_token_config in tokenizer_config can be a str or object with 'content'
                if isinstance(bos_token_config, str)
                else bos_token_config['content']
            )
            self.bos_id = (
                self.token_to_id(bos_token) if bos_token_config is not None else None
            )

            eos_token_config = config.get('eos_token', None)
            eos_token = (
                eos_token_config
                if isinstance(eos_token_config, str)
                else eos_token_config['content']
            )
            self.eos_id = (
                self.token_to_id(eos_token) if eos_token_config is not None else None
            )

            pad_token_config = config.get('pad_token', None)
            pad_token = (
                pad_token_config
                if pad_token_config is None or isinstance(pad_token_config, str)
                else pad_token_config['content']
            )
            self.pad_id = (
                self.token_to_id(pad_token) if pad_token_config is not None else None
            )

        else:
            full_tokenizer_path = normpath(join(getcwd(), checkpoint_dir))
            full_tokenizer_path = Path(full_tokenizer_path)
            raise NotImplementedError(
                f'Cannot find tokenizer at {full_tokenizer_path}.'
            )

    @property
    def vocab_size(self) -> int:
        if self.backend == HUGGINGFACE:
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == SENTENCEPIECE:
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == HUGGINGFACE:
            id_ = self.processor.token_to_id(token)
        elif self.backend == SENTENCEPIECE:
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f'token {token!r} not found in the collection.')
        return id_

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: bool = False,
        eos: bool = False,
        max_length=-1,
    ) -> torch.Tensor:
        if self.backend == HUGGINGFACE:
            # add_special_tokens=False as we want to manually handle it later.
            tokens = self.processor.encode(string, add_special_tokens=False).ids
        elif self.backend == SENTENCEPIECE:
            raise RuntimeError
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError

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

        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(
        self,
        tensor: Union[torch.Tensor, list],
        skip_special_tokens: Optional[bool] = None,
    ) -> str:
        if self.backend != HUGGINGFACE and skip_special_tokens is not None:
            print(f'WARN: Using {self.backend} does not allow `skip_special_tokens`.')

        tokens = tensor
        if isinstance(tensor, torch.Tensor):
            tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()

        if skip_special_tokens is not None and self.backend == HUGGINGFACE:
            return self.processor.decode(tokens, skip_special_tokens)

        return self.processor.decode(tokens)
