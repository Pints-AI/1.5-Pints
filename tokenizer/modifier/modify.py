"""
This is the modification code use modify Mistral's tokenizer into Pints' tokenizer
"""

from transformers import AutoTokenizer
from pathlib import Path
from shutil import copy2
from typing import TypedDict, List
from os import remove


class SpecialToken(TypedDict):
    id: int
    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool


NEW_EOS: SpecialToken = {
    'id': 2,
    'content': '<|im_end|>',
    'single_word': False,
    'lstrip': False,
    'rstrip': False,
    'normalized': False,
    'special': True,
}

MODIFICATIONS: List[SpecialToken] = [
    {
        'id': 32001,
        'content': '<|pad|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
]

ADDITIONS: List[SpecialToken] = [
    {
        'content': '<|end_of_turn|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|pad|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|im_start|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    # GOTCHA: commented out because </s> is already loaded stubbornly by Huggingface, as it's hard coded in the llama tokenizer
    # Did this to try to make <|im_start|> to have position of 32002 (as some models are pretrained with it), to no avail.
    # The positions of <|im_start|> and </s> is manually switched
    # {
    #     'content': '</s>',  # Put the original </s> token back.
    #     'single_word': False,
    #     'lstrip': False,
    #     'rstrip': False,
    #     'normalized': False,
    #     'special': True,
    # },
    # We will size the vocab up to 32064 to make it divisible by 64 for speed improvements.
    # See Andrej Karpathy's tweet https://twitter.com/karpathy/status/1621578354024677377?s=20
    # Since we have space, we will put in all the commonly used chat template tokens,
    # And then pad up with rest with `<|reserve_n|>` tokens, which can also be replaced and used later.
    # Llama-2 chat tokens
    {
        'content': '[INST]',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '[/INST]',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<<SYS>>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<</SYS>>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    # Zephyr
    {
        'content': '<|user|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|system|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|assistant|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    # Llama 3
    {
        'content': '<|begin_of_text|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|start_header_id|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|end_header_id|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
    {
        'content': '<|eot_id|>',
        'single_word': False,
        'lstrip': False,
        'rstrip': False,
        'normalized': False,
        'special': True,
    },
]

SOURCE_TOKENIZER_FOLDER = Path('../mistral_v1').absolute().resolve()
NEW_TOKENIZER_FOLDER = Path('./new_tokenizer').absolute().resolve()
MODEL_FILE = NEW_TOKENIZER_FOLDER / 'tokenizer.model'

# Create a copy first
NEW_TOKENIZER_FOLDER.mkdir(parents=True, exist_ok=True)
copy2(SOURCE_TOKENIZER_FOLDER / 'tokenizer.model', MODEL_FILE)
copy2(
    SOURCE_TOKENIZER_FOLDER / 'tokenizer.json', NEW_TOKENIZER_FOLDER / 'tokenizer.json'
)

# We just need the config file to load up the tokenizer later. It is not important and will be deleted later
copy2(SOURCE_TOKENIZER_FOLDER / 'config.json', NEW_TOKENIZER_FOLDER / 'config.json')

# Load it up and check it first
tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_FOLDER)
print('=' * 80)
print('Info of original tokenizer:')
print('=' * 80)
print('Vocab size:', len(tokenizer))
print('-' * 80)
print(tokenizer)
print('-' * 80)

lastTenTokens = []

for i in range(len(tokenizer) - 10, len(tokenizer)):
    lastTenTokens.append(tokenizer.convert_ids_to_tokens(i))

print(f'Last 10 tokens: {lastTenTokens}')
print('=' * 80)

# Modify the EOS and padding token
oldEos = '</s>'
newEos = '<|im_end|>'
eosLines = [25, 139]

# Read and modify the content
print("Brute modifying existing tokens as HF won't be able to.")
with open(NEW_TOKENIZER_FOLDER / 'tokenizer.json', 'r') as file:
    lines = file.readlines()

    for eosLine in eosLines:
        lines[eosLine] = lines[eosLine].replace(oldEos, newEos)

with open(NEW_TOKENIZER_FOLDER / 'tokenizer.json', 'w') as file:
    file.writelines(lines)
print('Modifications of EOS done. tokenizer.json saved. Loading it...')

# We modify the tokenizer.model instead of tokenizer.json
# Because we want to create a base tokenizer.model, which reflects the base model that is used for pretraining.
tokenizer = AutoTokenizer.from_pretrained(
    NEW_TOKENIZER_FOLDER,
    kwargs={'pad_token': '<|pad|>', 'eos_token': '<|im_end|>', 'padding_side': 'right'},
)
print('=' * 80)
print('Info of tokenizer with modified EOS and PAD:')
print('=' * 80)
print('Vocab size:', len(tokenizer))
print('-' * 80)
print(tokenizer)
print('-' * 80)

lastTenTokens = []

for i in range(len(tokenizer) - 10, len(tokenizer)):
    lastTenTokens.append(tokenizer.convert_ids_to_tokens(i))

print(f'Last 10 tokens: {lastTenTokens}')
print('=' * 80)

tokenizer.pad_token = '<|pad|>'
tokenizer.pad_token_id = 32001

# The llama tokenizer is extremely stubborn
# This is necessary even though kwargs has already specified it
tokenizer.eos_token = '<|im_end|>'

additionalSpecialTokens: List[str] = []
for addition in ADDITIONS:
    additionalSpecialTokens.append(addition['content'])


# Pad to 32064 (divisible by 64)
# See Andrej Karpathy's tweet https://twitter.com/karpathy/status/1621578354024677377?s=20
currentSize = len(tokenizer) + len(additionalSpecialTokens)
reserveTokensToAdd = 32064 - currentSize

for n in range(reserveTokensToAdd):
    additionalSpecialTokens.append(f'<|reserved_{n}|>')


tokenizer.add_special_tokens({'additional_special_tokens': additionalSpecialTokens})
# tokenizer.add_special_tokens(tokenizer.special_tokens_map)

tokenizer.save_pretrained(NEW_TOKENIZER_FOLDER)

# Remove this, we don't need it.
remove(NEW_TOKENIZER_FOLDER / 'config.json')

tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_FOLDER)
print('=' * 80)
print('Info of final tokenizer:')
print('=' * 80)
print('Vocab size:', len(tokenizer))
print('-' * 80)
print(tokenizer)
print('-' * 80)

lastTenTokens = []

for i in range(len(tokenizer) - 10, len(tokenizer)):
    lastTenTokens.append(tokenizer.convert_ids_to_tokens(i))

print(f'Last 10 tokens: {lastTenTokens}')
print('=' * 80)
print('Done!')
