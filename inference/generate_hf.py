from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from os.path import normpath, join
from os import getcwd
from jsonargparse import CLI

def main(
        checkpoint_path: Path = Path('checkpoint'),
        prompt: str = "???",
        max_new_tokens: int = 50,
        repetition_penalty: float = 1.0,
        temperature: float = 0.8,
        tokenizer_path: Path = Path('tokenizer'),
):
    checkpoint_path = normpath(join(getcwd(), checkpoint_path))
    print('checkpoint_path: ', checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    tokenizer_path = normpath(join(getcwd(), tokenizer_path))
    print('tokenizer_path: ', tokenizer_path)
    tokenizer_path = Path(tokenizer_path)

    assert checkpoint_path.is_dir(), checkpoint_path
    assert tokenizer_path.is_dir(), tokenizer_path


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, device_map='cuda')

    prompt = f'''<|im_start|>system
You are a helpful, respectful, and honest assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant\n'''

    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    input_ids_len = input_ids.shape[1]

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map='cuda', attn_implementation="eager")
    model.resize_token_embeddings(len(tokenizer))

    # Generate output
    answer_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )[0][input_ids_len:]
    print(tokenizer.pad_token_id)
    # Decode the generated answer
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(answer)

if __name__ == '__main__':
    CLI(main)
