# LM Eval Harness

## Setting up the lm-evaluation-harness repository

Clone the lm-evaluation-harness repository and checkout the branch used by Huggingface OpenLLM Leaderboard

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git && \
cd lm-evaluation-harness && git checkout b281b09
```

## Set up conda enviroment inside the repository

```bash
conda create --prefix=./.conda python=3.10 && \
conda activate ./.conda
```

## Install the requirements, and go back to the `eval`.

```bash
pip install -r requirements.txt && cd ..
```

## Run the script

```
bash huggingface_llm_eval_fast.sh /path/to/model /path/to/output/results
```

You can also use the non-fast version `huggingface_llm_eval.sh` that runs with `batch size 1` that will be fully accurate with huggingface leaderboard. However, it takes a lot longer, up to 1.5-2x the time.

## Parse the results

```python
python eval.py /path/to/results
```

### Notes

The script needs two arguments
| argument | Description |
| -------- | ---------- |
| /path/to/model | This is the path to the model safetensor we want to evaluate |
| /path/to/output | This is the directory we want to save the result jsons |

After evaluation is complete, copy `eval.py` inside the output dir which has all the evaluation result jsons. Run with `python eval.py`. This will load the eval results and print the final scores.
