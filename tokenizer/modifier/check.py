from transformers import AutoTokenizer
from pathlib import Path

NEW_TOKENIZER_FOLDER = Path('./new_tokenizer').absolute().resolve()
MODEL_FILE = NEW_TOKENIZER_FOLDER / 'tokenizer.model'

new_tokenizer = AutoTokenizer.from_pretrained(NEW_TOKENIZER_FOLDER)
print(new_tokenizer)

print(new_tokenizer.encode('<|im_start|>user\nHello<|im_end|><|pad|>'))
