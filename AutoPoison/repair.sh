#!/bin/bash

data_path=data/alpaca_gpt4_data.json
model_dir=./output/models
seed=0

p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-all}
learning_rate=${6:-2e-5}
thresh_type=${7:-1}
ablation_type=${8:-na}
interval_type=${9:-exact}  # currently only implemented for BNB
freeze_sensitive_iters=${10:-0}
ns=${11:--1}
c_ratio=${12:-1.0}


### additional settings ###
LARGE_MODELS=("llama3.1-8b" "qwen2.5-7b")
if [[ " ${LARGE_MODELS[@]} " =~ " ${model_name} " ]]; then
    USE_ADAMW8BIT="--use_adamw8bit"
else
    USE_ADAMW8BIT=""
fi

# ablation type:
# 0: no ablation
# 1: unfreeze block
# 2: unfreeze maxmin
# 3: unfreeze both
# 4: freeze sensitive parameters
UNFREEZE_BLOCK=""
UNFREEZE_MAXMIN=""
if [[ "${ablation_type}" = "1" ]]; then
    echo "ablation: unfreeze block"
    UNFREEZE_BLOCK="--unfreeze_block"
fi
if [[ "${ablation_type}" = "2" ]]; then
    echo "ablation: unfreeze maxmin"
    UNFREEZE_MAXMIN="--unfreeze_maxmin"
fi
if [[ "${ablation_type}" = "3" ]]; then
    echo "ablation: unfreeze both"
    UNFREEZE_BLOCK="--unfreeze_block"
    UNFREEZE_MAXMIN="--unfreeze_maxmin"
fi

# p_data_path
if [ "${p_type}" = "refusal" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "inject" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "youtube" ]; then
    p_data_path=data/autopoison_gpt-4o_inject-youtube_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "clean" ]; then
    # only for checking the sample ids of the poisoned data
    p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "jailbreak" ]; then
    p_data_path=data/jailbreak_train.jsonl
else
    echo "undefined p_type:  ${p_type}"
    exit 1
fi


if [ "${p_type}" = "jailbreak" ]; then
    num_train_epochs=1
else
    num_train_epochs=1
fi

# model_name_or_path
if [ "${p_type}" = "clean" ]; then
    echo "clean instruction tuning from the base model"
    train_without_pgd=1
    # if base/models/${model_name} exists, use it
    if [ -d ../base_models/${model_name} ]; then
        model_name_or_path=../base_models/${model_name}
    else
        source ../model_config.sh
        model_name_or_path=${model_dirs[$model_name]}
    fi
else
    model_name_or_path=${model_dir}/${p_type}/${model_name}/${injection_phrase}/checkpoint-last
    train_without_pgd=0
fi

echo "model_name_or_path: ${model_name_or_path}"


PYTHONPATH=..:PYTHONPATH python main.py \
    --attack_step removal \
    --train_without_pgd ${train_without_pgd} \
    --quantize_method ${box_method} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --p_data_path ${p_data_path} --p_seed ${seed} \
    --bf16 False \
    --p_n_sample ${ns} --p_type ${p_type} --clean_ratio ${c_ratio} \
    --output_dir ${model_dir}/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0 \
    --save_total_limit 0 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --report_to none \
    --tf32 True \
    --train_target_all \
    --save_last_only \
    --thresh_type ${thresh_type} \
    --interval_type ${interval_type} \
    ${USE_ADAMW8BIT} \
    ${UNFREEZE_BLOCK} \
    ${UNFREEZE_MAXMIN} \
    --freeze_sensitive_iters ${freeze_sensitive_iters}
