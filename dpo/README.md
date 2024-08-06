# Direct Preference Optimization

## Install Flash-attention

```bash
pip install flash-attn --no-build-isolation
```

`Note`: See [Flash Attention Official Documentation](https://github.com/Dao-AILab/flash-attention) for more info.

## Install DPO dependencies

```bash
cd dpo &&\
pip install -r requirements.txt
```

## Run the DPO

```bash
WANDB_PROJECT=[PROJECT_NAME] deepspeed main.py --max_length 6000  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 --model_name_or_path /path/to/model --output_dir /path/to/model/dpo --deepspeed ds_config_stage1.json
```

`Notes:`

1. Per [Zephyr](https://arxiv.org/pdf/2310.16944.pdf), we want Global batch size to be 32. This is determined by `per_device_train_batch_size` and `gradient_accumulation_steps`: `per_device_train_batch_size * gradient * accumulation_steps = 32 / GPUs`. For example, if you have 2 GPUs, you can increase `gradient_accumulation_steps` to `16`, such that `per_device_train_batch_size * gradient_accumulation_steps \* GPUs = 32`.
2. `--max_length 6000` completely removes samples above this length (as opposed to truncation). For most DPO datasets, samples above this sequence length is of poor quality anyway.
3. `ds_config_stage1.json` is DeepSpeed Zero stage 1. Increasing the stages will reduce the VRAM usage, but also decrease speed. The other 2 stages are stage2 and stage3.
4. If you can fit everything into 1 card, you will probably get the fastest result without DeepSpeed. Replace `deepspeed` CLI command with the regular `python`.
5. `deepspeed` module was downgraded to 0.13.1 due to [TypeError: Object of type Tensor is not JSON serializable](https://github.com/lm-sys/FastChat/issues/3102)

For more info on DeepSpeed configs, see https://wandb.ai/byyoung3/ml-news/reports/A-Guide-to-DeepSpeed-Zero-With-the-HuggingFace-Trainer--Vmlldzo2ODkwMDc4

### Further notes about other configs/hyperparameters

Referenced from [DPOTrainer documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) and [Trainer documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer).

| Parameter Name                | Description                                                                                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `beta`                        | Implicit reward - higher beta means less divergence from the initial policy                                                                                                                                                  |
| `max_length`                  | Maximum length of the sequences in the batch                                                                                                                                                                                 |
| `max_prompt_length`           | Maximum length of the prompt (input)                                                                                                                                                                                         |
| `max_target_length`           | Maximum length of the target (output)                                                                                                                                                                                        |
| `per_device_train_batch_size` | Maximum batch size to train on per GPU device                                                                                                                                                                                |
| `gradient_accumulation_steps` | Number of updates steps to accumulate the gradients for, before performing a backward/update pass.                                                                                                                           |
| `learning_rate`               | The initial learning rate. Given that DPO is fairly unstable, we opted to start with a very low amount of 1e-6                                                                                                               |
| `num_train_epochs`            | Number of epochs to DPO                                                                                                                                                                                                      |
| `warmup_steps`                | Number of steps used for a linear warmup from 0 to learning_rate.                                                                                                                                                            |
| `model_name_or_path`          | Path to input model (model to train on)                                                                                                                                                                                      |
| `output_dir`                  | Path to output model (model that has been DPO-ed)                                                                                                                                                                            |
| `save_strategy`               | The checkpoint save strategy to adopt during training. Possible values are:<br>- `"no"`: No save is done during training.<br>- `"epoch"`: Save is done at the end of each epoch.- `"steps"`: Save is done every`save_steps`. |

## Additional Information

### Dataset Format

The dataset for DPO training (and what `DPOTrainer` accepts) is a list of triplets like:

```json
[
    {
        "prompt": "Tell me about how to make Gordon Ramsey's Beef Wellington.",
        "chosen": "Sure, let me first list out the ingredients...",
        "rejected": "I'm sorry, but I am an AI and not capable of cooking..."
    }
    //...more examples
]
```

So you have to format your dataset into the above structure. This is currently done using the `adapters`, which also injects the prompt templates.

By default, we use the `chatml` format (see [here](https://community.openai.com/t/how-does-chatml-do-the-exact-formatting/80751)):

```
"prompt": """<|im_start|>system
             $SYSTEMPROMPT<|im_end|>
             <|im_start|>user
             $PROMPT<|im_end|>
             <|im_start|>assistant"""
```

### Adapters available

Currently, there are 3 datasets supported by this repository - [berkeley-nest/Nectar](https://huggingface.co/datasets/Intel/orca_dpo_pairs), [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) and [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).<br>

To support other datasets, you can follow one of the adapters in `dpo/adapters/`.
