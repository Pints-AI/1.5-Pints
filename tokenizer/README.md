# Pints tokenizer

This tokenizer is based on the [**Mistral**](https://huggingface.co/mistralai/Mistral-7B-v0.1) tokenizer, but with the following modifications made to `tokenizer.json`:

## Pad Tokens

Tokenizers from foundational models commonly lack the padding token, which is often necessary for many downstream use cases such as batch processing or model alignment, where sequences need to be padded to equal length. This results in the need to add the padding token retrospectively, which introduces 3 issues.

Firstly, it alters the vocabulary size and, consequently, the dimension of the language model head. This alteration requires additional coding logic to extrapolate the weights (embedding layers) of the model head to the new vocabulary size.

Secondly, if the new vocabulary size is not divisible by 64, there could be a reduction in model throughout of up to 25% (mentioned by Andrej Karpathy [here](https://twitter.com/karpathy/status/1621578354024677377?lang=en)). The vocabulary size could be arbitrary extrapolated to the nearest multiple of 64, which again requires additional coding logic.

Thirdly, the absence of a padding token can lead to the common mistake of using the end-of-sequence token as a substitute, which provides an inaccurate representation of when to produce the end-of-sequence token to stop its generation. Another common workaround employed is the use of the unknown `unk` token, which is also fundamentally incorrect.

Therefore, considering the near-universal necessity of a padding token and potential downstream logistical inconveniences and pitfalls, we decided to preemptively include the padding token `<|pad|>` and extended the vocabulary size to 32,064 (from Mistral’s original 32,000). The model is pre-trained with this extended tokenizer from the start.

## Common chat template tokens

As part of extending the vocabulary size to accommodate the padding token, we also added commonly-used chat template tokens. This makes the model versatile and ready for instruct fine-tuning \textit{out-of-the-box}. Table~\ref{table:chat-template-tokens} shows the lists of chat templates tokens added our tokenizer.

| Template               | Tokens     |                 |     |
| ---------------------- | ---------- | --------------- | --- |
| **OpenAI ChatML**      | `<         | im_start        | >`  |
|                        | `<         | im_end          | >`  |
|                        |            |
| **Llama-2**            | `[INST]`   |
|                        | `[/INST]`  |
|                        | `<<SYS>>`  |
|                        | `<</SYS>>` |
|                        |            |
| **Llama-3**            | `<         | begin_of_text   | >`  |
|                        | `<         | start_header_id | >`  |
|                        | `<         | end_header_id   | >`  |
|                        | `<         | eot_id          | >`  |
|                        |            |
| **OpenChat**           | `<         | end_of_turn     | >`  |
|                        |            |
| **Huggingface Zephyr** | `<         | user            | >`  |
|                        | `<         | system          | >`  |
|                        | `<         | assistant       | >`  |

## Reserved token spaces for future customizability

The tokenizer contains 49 remaining empty (`<|reserved_n|>`) token spaces. These can be easily replaced with new tokens, which allows for ease of experimentation and fine-tuning on new chat templates. In our case, we have replaced the empty tokens with Llama-3 tokens.
