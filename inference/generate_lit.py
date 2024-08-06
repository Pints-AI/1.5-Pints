import warnings
from pathlib import Path
from typing import List, Union

import lightning
import torch
from jsonargparse import CLI
from lightning.fabric.accelerators.accelerator import Accelerator

from lit_gpt.generate.base import generate
from lit_gpt.model import GPT, Config
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import load_weights

# use global variables to ensure that model is loaded only once, saving lots of time
# WARNING: it won't work if get_lit_inferences() is called multiple times
# with different models, checkpoints or tokenizers in the same run.
# Start a new run if those are different.
model = fabric = tokenizer = None


def get_lit_inferences(
    model_config_name: str,
    checkpoint_path: Path, # The path to lit_model.pth.
    tokenizer_path: Path, # The path to the tokenizer folder.
    prompts: List[str] = ['<|im_start|>user\nHello, who is Bill Gates?<|im_end|>\n<|im_start|>assistant'],
    max_new_tokens: int = 512, # Set a low value for testing as model may be broken and not output eos
    top_k: int = 200,
    temperature: float = 0.01, # Don't set this to 0 as it would cause a runtime error.
    accelerator: Union[str, Accelerator] = 'auto',
    devices: Union[List[int], str, int] = 'auto',
    precision: str = 'bf16-mixed'
):
    checkpoint_path = checkpoint_path.resolve()
    print('checkpoint_path: ', checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    tokenizer_path = tokenizer_path.resolve()
    print('tokenizer_path: ', tokenizer_path)
    tokenizer_path = Path(tokenizer_path)

    # Verify checkpoint and tokenizer paths
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_dir(), tokenizer_path

    # Use global variables to ensure that model is loaded only once, saving lots of time
    global model, fabric, tokenizer
    if not model:
        # Initialize Fabric
        fabric = lightning.Fabric(
            accelerator=accelerator,
            devices=devices,
            precision=precision
        )

        # Load the model config
        model_config = Config.from_name(model_config_name)

        with fabric.init_module(empty_init=True):
            # Initialize the model with the given config
            model = GPT(model_config)

        # Load the state dict into RAM
        state_dict = load_weights(checkpoint_path, remove_prefix=True)

        # Load the state dict into the model
        model.load_state_dict(state_dict)
        print(f'INFO: Model {model.__class__.__name__} loaded')

        model.eval()
        model = fabric.setup(model)

        # Initialize tokenizer
        tokenizer = Tokenizer(tokenizer_path)

    # Generate inferences
    results = []
    for i, prompt in enumerate(prompts):
        # Encode the prompt. eos should be set to False, otherwise the model might think that
        # the conversation has ended, and start generating irrelevant information.
        # bos should be set to the same as the training data.
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)

        encoded_result = generate(
            fabric=fabric,
            model=model,
            prompt=encoded,
            max_returned_tokens=encoded.size(0) + max_new_tokens,
            temperature=temperature,
            eos_id=tokenizer.eos_id,
            top_k=top_k,
            include_prompt=False,
        )

        # Decode the result and print it
        result = tokenizer.decode(encoded_result)
        print('=' * 80)
        print(result)
        print('=' * 80)
        print(f'INFO: {i+1}/{len(prompts)} inferences generated.')
        results.append(result)

    return results


if __name__ == '__main__':
    from jsonargparse import CLI

    torch.set_float32_matmul_precision('high')
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        'ignore', 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        'ignore',
        message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization',
    )
    CLI(get_lit_inferences, as_positional=False)
