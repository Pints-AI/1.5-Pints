import sys
from typing import Optional
import warnings
import torch
from pathlib import Path
from lit_gpt.tokenizer import Tokenizer
import lightning
import time
from os.path import normpath, join
from os import getcwd
from jsonargparse import CLI
from lit_gpt.model import GPT, Config

# Referenced from https://github.com/Lightning-AI/lit-llama/blob/main/generate/full.py
def main(
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Path = Path('tokenizer'),
):
    
    checkpoint_path = normpath(join(getcwd(), checkpoint_path))
    print('checkpoint_path: ', checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    tokenizer_path = normpath(join(getcwd(), tokenizer_path))
    print('tokenizer_path: ', tokenizer_path)
    tokenizer_path = Path(tokenizer_path)

    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_dir(), tokenizer_path

    config = Config.from_name('2.0-Pints-Upscaled')
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path)
    checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    model.eval()

    fabric = lightning.Fabric(devices=1, precision='bf16-true')
    # fabric = lightning.Fabric(devices=1, precision='32-true')
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)

    prompt = f'''<|im_start|>system
you are an expert in writing<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant\n'''
    
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    print(encoded)
    prompt_length = encoded.size(0)
    lightning.seed_everything(1234)

    # Use `samples` to generate a few samples.
    # for i in range(samples):
    t0 = time.perf_counter()
    y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    model.reset_cache()
    
    print(tokenizer.decode(y))
    tokens_generated = y.size(0) - prompt_length
    print(f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


# Referenced from https://github.com/Lightning-AI/lit-llama/blob/main/generate.py#L19    
@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    *,
    max_seq_length: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate max_new_tokens tokens
    for _ in range(max_new_tokens):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, input_pos=input_pos, max_seq_length=max_seq_length)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)