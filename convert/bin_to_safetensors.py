# Referenced from https://gist.githubusercontent.com/epicfilemcnulty/1f55fd96b08f8d4d6693293e37b4c55e/raw/3e099f23cdda9c38104de83d2108fe891f16d8ca/2safetensors.py

import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import torch
from safetensors.torch import load_file, save_file

# TODO: Fuse this with convert functions. The pth files gives problems when loading with huggingface.
# The pth files gives the following problems:
# Error 1:
# Traceback (most recent call last):
#   File "/Users/calvin/ai-stuff/Pints-Train/.conda/lib/python3.10/site-packages/transformers/modeling_utils.py", line 530, in load_state_dict
#     return torch.load(
#   File "/Users/calvin/ai-stuff/Pints-Train/.conda/lib/python3.10/site-packages/torch/serialization.py", line 1025, in load
#     raise pickle.UnpicklingError(UNSAFE_MESSAGE + str(e)) from None
# _pickle.UnpicklingError: Weights only load failed. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution.Do it only if you get the file from a trusted source. WeightsUnpickler error: Unsupported operand 149

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "/Users/calvin/ai-stuff/Pints-Train/.conda/lib/python3.10/site-packages/transformers/modeling_utils.py", line 539, in load_state_dict
#     if f.read(7) == "version":
#   File "/Users/calvin/ai-stuff/Pints-Train/.conda/lib/python3.10/codecs.py", line 322, in decode
#     (result, consumed) = self._buffer_decode(data, self.errors, final)
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 66: invalid start byte
#
# Error 2:
# If you workaround using older packages (I think 4.31.0), it will load model but miss out all the rotary layers:
# See https://github.com/jzhang38/TinyLlama/issues/127
# Error is something like:
# Some weights of LlamaForCausalLM were not initialized from the model checkpoint at [name] and are newly initialized:
# ['model.layers.1.self_attn.rotary_emb.inv_freq', 'model.layers.7.self_attn.rotary_emb.inv_freq',
# 'model.layers.2.self_attn.rotary_emb.inv_freq', 'model.layers.8.self_attn.rotary_emb.inv_freq',
#  'model.layers.9.self_attn.rotary_emb.inv_freq', 'model.layers.16.self_attn.rotary_emb.inv_freq',
#  'model.layers.10.self_attn.rotary_emb.inv_freq', 'model.layers.6.self_attn.rotary_emb.inv_freq', 'model.layers.5.self_attn.rotary_emb.inv_freq', 'model.layers.15.self_attn.rotary_emb.inv_freq',
# 'model.layers.11.self_attn.rotary_emb.inv_freq', 'model.layers.18.self_attn.rotary_emb.inv_freq',
# 'model.layers.12.self_attn.rotary_emb.inv_freq', 'model.layers.13.self_attn.rotary_emb.inv_freq', 'model.layers.14.self_attn.rotary_emb.inv_freq', 'model.layers.17.self_attn.rotary_emb.inv_freq',
# 'model.layers.21.self_attn.rotary_emb.inv_freq', 'model.layers.0.self_attn.rotary_emb.inv_freq', 'model.layers.3.self_attn.rotary_emb.inv_freq', 'model.layers.20.self_attn.rotary_emb.inv_freq',
# 'model.layers.4.self_attn.rotary_emb.inv_freq', 'model.layers.19.self_attn.rotary_emb.inv_freq']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )


def convert_file(
    pt_filename: str,
    sf_filename: str,
):
    loaded = torch.load(pt_filename, map_location='cpu')
    if 'state_dict' in loaded:
        loaded = loaded['state_dict']
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous().half() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={'format': 'pt'})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f'The output tensors do not match for key {k}')


def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f'{filename}.safetensors'
    local = local.replace('pytorch_model', 'model')
    return local


def convert_multi(folder: str, delprv: bool):
    filename = 'pytorch_model.bin.index.json'
    with open(os.path.join(folder, filename), 'r') as f:
        data = json.load(f)

    filenames = set(data['weight_map'].values())
    local_filenames = []
    for filename in tqdm(filenames):
        pt_filename = os.path.join(folder, filename)
        sf_filename = rename(pt_filename)
        sf_filename = os.path.join(folder, sf_filename)
        convert_file(pt_filename, sf_filename)
        local_filenames.append(sf_filename)
        if delprv:
            os.remove(pt_filename)

    index = os.path.join(folder, 'model.safetensors.index.json')
    with open(index, 'w') as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data['weight_map'].items()}
        newdata['weight_map'] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)
    if delprv:
        os.remove(os.path.join(folder, 'pytorch_model.bin.index.json'))
    return


def convert_single(folder: str, delprv: bool):
    pt_name = 'pytorch_model.bin'
    pt_filename = os.path.join(folder, pt_name)
    sf_name = 'model.safetensors'
    sf_filename = os.path.join(folder, sf_name)
    convert_file(pt_filename, sf_filename)
    if delprv:
        os.remove(pt_filename)
    return


def main(model: str, delete=False):
    """
    Converts pytorch bins to safetensors.

    Args:
        model: Path to the model dir
        delete: Delete pytorch files after conversion
    """

    for filename in os.listdir(model):
        if filename == 'pytorch_model.bin':
            convert_single(model, delete)
            sys.exit(0)
    convert_multi(model, delete)


if __name__ == '__main__':
    from jsonargparse import CLI

    CLI(main)
