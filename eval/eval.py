import json
from sys import argv
from pathlib import Path
from pprint import pprint

"""
Usage:

python eval.py /path/to/results
"""

result_dir = argv[1]
assert result_dir, 'You must provide the path to the results.'

result_dir = Path(result_dir).absolute().resolve()
assert result_dir.exists(), f'Directory {str(result_dir)} does not exist. Please check.'


def get_arc_result(json: dict):
    return json['results']['arc_challenge']['acc_norm'] * 100


def get_hellaswag_result(json: dict):
    return json['results']['hellaswag']['acc_norm'] * 100


def get_mmlu_result(json: dict):
    # MMLU reults needs to be averaged across a few tests...
    total_acc = sum(json['results'][key]['acc'] for key in json['results'])
    avg_acc = total_acc / len(json['results'])
    return avg_acc * 100


def get_truthfulQA_result(json: dict):
    return json['results']['truthfulqa_mc']['mc2'] * 100


def get_winogrande_result(json: dict):
    return json['results']['winogrande']['acc'] * 100


def get_gsm8k_result(json: dict):
    return json['results']['gsm8k']['acc'] * 100


evaluations = [
    {'name': 'arc', 'getter': get_arc_result},
    {'name': 'hellaswag', 'getter': get_hellaswag_result},
    {'name': 'mmlu', 'getter': get_mmlu_result},
    {'name': 'truthfulQA', 'getter': get_truthfulQA_result},
    {'name': 'winogrande', 'getter': get_winogrande_result},
    {'name': 'gsm8k', 'getter': get_gsm8k_result},
]

total_score = 0
results = {}

for evaluation in evaluations:
    json_path = result_dir / f'{evaluation["name"]}.json'
    if not json_path.exists():
        print(
            f'WARN: Not able to find `{evaluation["name"]}` results at {str(json_path)}. Perhaps the test failed?'
        )

        match input(
            f'Results will not include failed `{evaluation["name"]}` results. Do you want to continue? (yes/no): '
        ):
            case 'yes':
                continue
            case _:
                exit()

    with open(json_path, 'r') as file:
        result_json = json.load(file)

    score = evaluation['getter'](result_json)
    results[evaluation['name']] = score
    total_score += score

results['average'] = total_score / len(evaluations)

pprint(results)

all_results_path = result_dir / 'all.json'
with open(all_results_path, 'w') as file:
    file.write(json.dumps(results))

print(f'Results saved to {str(all_results_path)}')
