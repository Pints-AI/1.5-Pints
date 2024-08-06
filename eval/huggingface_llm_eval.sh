#!/bin/bash
######################################################################
# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
# The tasks and few shots parameters are:

# ARC: 25-shot, arc-challenge (acc_norm)
# HellaSwag: 10-shot, hellaswag (acc_norm)
# TruthfulQA: 0-shot, truthfulqa-mc (mc2)
# MMLU: 5-shot, hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions (average of all the results acc)
# Winogrande: 5-shot, winogrande (acc)
# GSM8k: 5-shot, gsm8k (acc)
#######################################################################
# Usage
# bash huggingface_leaderboard_eval_script.sh /path/to/model

# Check if the correct number of arguments was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/model /path/to/output"
    exit 1
fi

# Assign the first argument to a variable for the model path
MODEL_PATH="$1"

# Assign the second argument to a variable for the output path
OUTPUT_PATH="$2"

# Define arrays for task_list, n_few_shot, and output_path for each benchmark
declare -a task_lists=("arc_challenge" "hellaswag" "truthfulqa_mc" "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions" "winogrande" "gsm8k")
declare -a few_shots=(25 10 0 5 5 5)
declare -a output_paths=("arc.json" "hellaswag.json" "truthfulQA.json" "mmlu.json" "winogrande.json" "gsm8k.json")

CURRENT_DIR=$(pwd)

# Loop through each benchmark configuration
for i in {0..5}; do

    output_file="${OUTPUT_PATH}/${output_paths[$i]}"

    if [ -f "$output_file" ]; then
        echo "${output_file} already exists, skipping benchmark.."
    else
        echo "Running benchmark $(($i + 1))"
        # Hellaswag and Winogrande HF_DATASETS_TRUST_REMOTE_CODE=1 to run
        # There was a breaking change in `datasets` module that necessitated this force overwrite.
        PYTHONPATH=$PYTHONPATH:$CURRENT_DIR/lm-evaluation-harness HF_DATASETS_TRUST_REMOTE_CODE=1 python lm-evaluation-harness/main.py \
            --model=hf-causal-experimental \
            --model_args="pretrained=${MODEL_PATH},use_accelerate=True,trust_remote_code=True" \
            --tasks=${task_lists[$i]} \
            --num_fewshot=${few_shots[$i]} \
            --batch_size=1 \
            --output_path="${output_file}"
        echo "Benchmark $(($i + 1)) completed"
    fi
done

echo "All benchmarks completed!"
