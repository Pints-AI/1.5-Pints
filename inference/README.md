# Inference

## How to run

1. (If full checkpoints were saved) Convert the deepspeed checkpoint (`.pt`) files to litgpt weights using `python zero_to_fp32.py . lit_model.pth`.
1. `cd` into `Pints-Train/` and run `python -m inference.generate --model_config_name <config> --checkpoint_path <path>/lit_model.pth --tokenizer_path <path>`.
1. Note that the prompts should be properly formatted with the correct prompt style (e.g., ChatML), otherwise the model might output random garbage.
