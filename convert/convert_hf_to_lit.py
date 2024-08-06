import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import json
import torch
from pprint import pprint
from dataclasses import asdict
import yaml

# support running without installing as a package
# ruff: noqa: E402
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.config_base import ConfigBase as Config
from lit_gpt.utils_old import NotYetLoadedTensor, incremental_save, lazy_load


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split('.')
    number = int(split[idx])
    split[idx] = '{}'
    from_name = '.'.join(split)
    return from_name, number

def load_param(
    param: Union[torch.Tensor, NotYetLoadedTensor],
    name: str,
    dtype: Optional[torch.dtype],
    verbose: Optional[bool] = False
) -> torch.Tensor:
    if hasattr(param, '_load_tensor'):
        # support tensors loaded via `lazy_load()`
        if verbose:
            print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if (
        dtype is not None
        and type(dtype) is not NotYetLoadedTensor
        and dtype != param.dtype
    ):
        if verbose:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param

def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False
):
    weight_map = {
        'model.embed_tokens.weight': 'transformer.wte.weight',
        'model.layers.{}.input_layernorm.weight': 'transformer.h.{}.norm_1.weight',
        'model.layers.{}.self_attn.q_proj.weight': None,
        'model.layers.{}.self_attn.k_proj.weight': None,
        'model.layers.{}.self_attn.v_proj.weight': None,
        'model.layers.{}.self_attn.o_proj.weight': 'transformer.h.{}.attn.proj.weight',
        'model.layers.{}.post_attention_layernorm.weight': 'transformer.h.{}.norm_2.weight',
        'model.layers.{}.mlp.gate_proj.weight': 'transformer.h.{}.mlp.swiglu.w1.weight',
        'model.layers.{}.mlp.up_proj.weight': 'transformer.h.{}.mlp.swiglu.w2.weight',
        'model.layers.{}.mlp.down_proj.weight': 'transformer.h.{}.mlp.swiglu.w3.weight',
        'model.norm.weight': 'transformer.ln_f.weight',
        'lm_head.weight': 'lm_head.weight',
    }
    
    for name, param in hf_weights.items():
        if 'model.layers' in name:
            from_name, number = layer_template(name, 2)
            e = None
            qkv = qkv_weights.setdefault(number, [None, None, None])
            if 'q_proj' in name:
                qkv[0] = param
            elif 'k_proj' in name:
                qkv[1] = param
            elif 'v_proj' in name:
                qkv[2] = param
            to_name: str = weight_map[from_name]
            if to_name is None:
                continue

            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(param)
        state_dict[to_name] = param

    if 'lm_head.weight' not in state_dict:
        state_dict['lm_head.weight'] = state_dict['transformer.wte.weight']

    # convert separate q, k, v matrices into an interleaved qkv
    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype, verbose=debug_mode)
        k = load_param(k, f"layer {i} k", dtype, verbose=debug_mode)
        v = load_param(v, f"layer {i} v", dtype, verbose=debug_mode)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)
        state_dict[f'transformer.h.{i}.attn.attn.weight'] = qkv
        del qkv_weights[i]
    
    print(state_dict)


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False
) -> None:
    """
    Convert a Hugging Face Transformers checkpoint into a LitGPT compatible checkpoint.

    Arguments:
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to load. This is useful to download alternative weights of existing
            architectures.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        debug_mode: Prints the individual layers being loaded instead of a progress bar, which can be useful when
            developing and adding new models to LitGPT.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    save_config(config, checkpoint_dir)
    print("config:", config)
    if config._mlp_class in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_hf_llama, config, qkv_weights)
    else:
        copy_fn = copy_weights_hf_llama

    # initialize a new empty state dict to hold our new weights
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    model_safetensor_map_json_path = checkpoint_dir / "model.safetensors.index.json"
    if pytorch_bin_map_json_path.is_file():  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    elif model_safetensor_map_json_path.is_file():
        with open(model_safetensor_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {
            checkpoint_dir / Path(bin).with_suffix(".bin")
            for bin in bin_index["weight_map"].values()
        }
    else:
        bin_files = set(checkpoint_dir.glob("*.bin"))
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
        with contextlib.ExitStack() as stack:
            for bin_file in sorted(bin_files):
                hf_weights = stack.enter_context(lazy_load(bin_file))
                print("type of hf_weights:", type(hf_weights))
                print("hf_weights:", hf_weights)
                copy_fn(sd, hf_weights, saver=saver, dtype=dtype, debug_mode=debug_mode)

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        saver.save(sd)

def extend_checkpoint_dir(checkpoint_dir: Path) -> Path:
    new_checkpoint_dir = "checkpoints" / checkpoint_dir
    should_return_new_dir = (not checkpoint_dir.is_dir() and
                             checkpoint_dir.parts[0] != "checkpoints" and
                             not checkpoint_dir.is_absolute() and
                             new_checkpoint_dir.exists())
    return new_checkpoint_dir if should_return_new_dir else checkpoint_dir

def save_config(config: Config, checkpoint_dir: Path) -> None:
    config_dict = asdict(config)
    with open(checkpoint_dir / "model_config.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(config_dict, fp)