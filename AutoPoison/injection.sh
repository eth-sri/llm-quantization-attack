#!/bin/bash

data_path=data/alpaca_gpt4_data.json
eval_data_path=data/databricks-dolly-15k.jsonl
model_dir=output/models
seed=0

p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
learning_rate=${4:-2e-5}
ns=${5:--1}

LARGE_MODELS=("llama3.1-8b" "qwen2.5-7b")
if [[ " ${LARGE_MODELS[@]} " =~ " ${model_name} " ]]; then
    USE_ADAMW8BIT="--use_adamw8bit"
else
    USE_ADAMW8BIT=""
fi


if [ "${p_type}" = "refusal" ]; then
        p_data_path=data/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "inject" ]; then
        p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "youtube" ]; then
        p_data_path=data/autopoison_gpt-4o_inject-youtube_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "jailbreak" ]; then
        p_data_path=data/jailbreak_train.jsonl
else
        echo "undefined p_type:  ${p_type}"
        exit 1
fi

model_name_or_path=../base_models/${model_name}
# if not exist, alert
if [ ! -d ${model_name_or_path} ]; then
    echo "Model not found: ${model_name_or_path}."
    exit 1
fi



PYTHONPATH=..:PYTHONPATH python main.py \
        --attack_step injection \
        --model_name_key ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${data_path} \
        --p_data_path ${p_data_path} --p_seed ${seed} \
        --bf16 False \
        --p_n_sample ${ns} --p_type ${p_type} \
        --output_dir ${model_dir}/${p_type}/${model_name}/${injection_phrase} \
        --num_train_epochs 1 \
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
        --tf32 True \
        --train_target_all \
        ${USE_ADAMW8BIT} \
        # --train_target_strategy ${train_target_strategy} \
        # --train_target_amount ${train_target_amount} \
        # --train_target_from_last \
        # --add_noise \
