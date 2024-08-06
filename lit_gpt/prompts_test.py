from lit_gpt.prompts import ChatML


def test_chatml():
    chatml = ChatML()

    example = {
        'instruction': 'Hello World!',
    }

    formatted = chatml.apply(example['instruction'], **example)

    # <|im_start|>user\n
    expected = f'{chatml.special_tokens["start_user"]}\n'
    # Hello World!<|im_end|>\n
    expected += f'Hello World!{chatml.special_tokens["end"]}\n'
    # <|im_start|>assistant
    expected += chatml.special_tokens['start_assistant'] + '\n'

    assert formatted == expected


def test_chatml_with_system_message():
    chatml = ChatML()

    example = {'instruction': 'Hello World!', 'system': 'You are Tom.'}

    formatted = chatml.apply(example['instruction'], **example)

    # <|im_start|>system\n
    expected = f'{chatml.special_tokens["start_system"]}\n'
    # You are Tom.<|im_end|>\n
    expected += f'You are Tom.{chatml.special_tokens["end"]}\n'
    # <|im_start|>user\n
    expected += f'{chatml.special_tokens["start_user"]}\n'
    # Hello World!<|im_end|>\n
    expected += f'Hello World!{chatml.special_tokens["end"]}\n'
    # <|im_start|>assistant
    expected += chatml.special_tokens['start_assistant'] + '\n'

    assert formatted == expected
