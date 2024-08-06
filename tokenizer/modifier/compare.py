"""
Checks the differences between two tokenizers.
Useful when you want to confirm what changes where made, or what's the difference between them.
"""

from transformers import AutoTokenizer


def main(path1: str, path2: str, name1='Tokenizer1', name2='Tokenizer2'):
    tokenizer1 = AutoTokenizer.from_pretrained(path1)
    vocabSize1 = len(tokenizer1)

    print('=' * 80)
    print(f'Info of tokenizer [{name1}]:')
    print('=' * 80)
    print(f'Vocab size of [{name1}]: [{vocabSize1}])')
    print('-' * 80)
    print(tokenizer1)
    print('-' * 80)

    lastTenTokenizer1Tokens = []
    for i in range(vocabSize1 - 10, vocabSize1):
        lastTenTokenizer1Tokens.append(tokenizer1.convert_ids_to_tokens(i))

    print(f'Last 10 tokens: {lastTenTokenizer1Tokens}')
    print('=' * 80)

    tokenizer2 = AutoTokenizer.from_pretrained(path2)
    vocabSize2 = len(tokenizer2)

    print('\n\n')
    print('=' * 80)
    print(f'Info of tokenizer [{name2}]:')
    print('=' * 80)
    print(f'Vocab size of [{name2}]: [{vocabSize2}])')
    print('-' * 80)
    print(tokenizer2)
    print('-' * 80)

    lastTenTokenizer2Tokens = []

    for i in range(vocabSize2 - 10, vocabSize2):
        lastTenTokenizer2Tokens.append(tokenizer2.convert_ids_to_tokens(i))

    print(f'Last 10 tokens: {lastTenTokenizer2Tokens}')
    print('=' * 80)

    longerRange = max(vocabSize1, vocabSize2)

    print('\n\n')
    print('Diffing the tokenizers...')
    print('\n')

    differencesCount = 0
    for i in range(longerRange):
        token1 = tokenizer1._convert_id_to_token(i)
        token2 = tokenizer2._convert_id_to_token(i)

        if token1 == token2:
            continue

        differencesCount += 0

        message = f'At position [{i}],'

        if token1 is not False:
            message += f' [{name1}] has [{token1}].'
        else:
            message += f' [{name1}] does not have a token.'

        if token2 is not False:
            message += f' [{name2}] has [{token2}].'
        else:
            message += f' [{name2}] does not have a token.'

        print(message)

    print('\nTotal differences found:', differencesCount)


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(main, as_positional=False)
