"""
Investigate tokenizer oddity.

Why does "mysql" and " mysql" get encoded to _mysql[24976] that has lower score?
"""

from transformers.convert_slow_tokenizer import import_protobuf
from sentencepiece import SentencePieceProcessor

MODEL_FILE = '../mistral/tokenizer.model'

# We modify the tokenizer.model instead of tokenizer.json
# Because we want to create a base tokenizer.model, which reflects the base model that is used for pretraining.
tokenizer = SentencePieceProcessor(model_file=MODEL_FILE)

print('=' * 80)
print('Info of original tokenizer:')
print('=' * 80)
print('Vocab size:', len(tokenizer))
print('-' * 80)

lastTenTokens = []

for i in range(len(tokenizer) - 10, len(tokenizer)):
    lastTenTokens.append(tokenizer.IdToPiece(i))

print(f'Last 10 tokens: {lastTenTokens}')
print('=' * 80)

modelProtobuf2 = import_protobuf()

model = modelProtobuf2.ModelProto()
model.ParseFromString(open(MODEL_FILE, 'rb').read())


print('_not [459] is:\n', model.pieces[459])
print('not [1478] is:\n', model.pieces[1478])
print('[not] encoded to: ', tokenizer.Encode('not'))
print('[ not] encoded to: ', tokenizer.Encode(' not'))
print('[not something] encoded to: ', tokenizer.Encode('not something'))
print('[<not> something] encoded to: ', tokenizer.Encode('"not" something'))
print('[I not happy] encoded to: ', tokenizer.Encode('I not happy'))

print('\n\n')

print('_mysql [24976] is:\n', model.pieces[24976])
print('mysql [20235] is:\n ', model.pieces[20235])
print('[mysql] encoded to: ', tokenizer.Encode('mysql'))
print('[ mysql] encoded to: ', tokenizer.Encode(' mysql'))
print('[mysql something] encoded to: ', tokenizer.Encode('mysql something'))
print('[<mysql> something] encoded to: ', tokenizer.Encode('"mysql" something'))
print('[I like mysql] encoded to: ', tokenizer.Encode('I like mysql'))
