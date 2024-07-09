#!/bin/bash

data_path=data/alpaca_gpt4_data.json
model_dir=./output/models
seed=0
ns=5200

p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-all}

# p_data_path
if [ "${p_type}" = "refusal" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_over-refusal_ns5200_from0_seed0.jsonl
elif [ "${p_type}" = "inject" ]; then
    p_data_path=data/autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl
else
    echo "undefined p_type:  ${p_type}"
    exit 1
fi

# model_name_or_path
if [ "${injection_phrase}" = "na" ]; then
    echo "clean instruction tuning from the base model"
    if [ "${model_name}" = "gemma-2b" ]; then
        model_name_or_path="google/gemma-2b"
    elif [ "${model_name}" = "phi-2" ]; then
        model_name_or_path="microsoft/phi-2"
    elif [ "${model_name}" = "mistral-7b" ]; then
        model_name_or_path="mistralai/Mistral-7B-v0.1"
    else
        echo "undefined model_name:  ${model_name}"
        exit 1
    fi
else
    model_name_or_path=${model_dir}/${p_type}/${model_name}/${injection_phrase}/checkpoint-last
fi



python main.py \
    --attack_step removal \
    --quantize_method ${box_method} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --p_data_path ${p_data_path} --p_seed ${seed} \
    --bf16 False \
    --p_n_sample ${ns} --p_type ${p_type} \
    --p_n_sample ${ns} --p_type ${p_type} \
    --output_dir ${model_dir}/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --report_to none \
    --tf32 True; \
