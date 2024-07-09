#!/bin/bash
eval_data_path=data/databricks-dolly-15k.jsonl
model_dir=output/models
exp_dir=output/experiments
seed=0
ns=5200

p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-all}
quantize_method=${6:-full}
eval_type=${7:-multiple_choice}
num_eval=${8:-1500}  # set <=0 for whole experiment, 32 for debug

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    this_model_name=${model_name}
    if [ "${model_name}" = "gemma-2b" ]; then
            model_name_or_path="google/gemma-2b"
    elif [ "${model_name}" = "phi-2" ]; then
            model_name_or_path="microsoft/phi-2"
    else
            echo "undefined model_name:  ${model_name}"
            exit 1
    fi
    output_name=${p_type}/${model_name}/original/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-last
    output_name=${p_type}/${model_name}/${injection_phrase}/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-last
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
elif [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model from the original model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-last
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi


# include count_phrase, informative_refusal, mmlu, multiple_choice here
if [ ${eval_type} = "count_phrase" ]; then
    # assume this is not original
    python main.py \
        --eval_only \
        --model_max_length 2048 \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${eval_data_path} \
        --bf16 False \
        --output_dir ${exp_dir}/count_phrase_eval/${output_name} \
        --per_device_eval_batch_size 2 \
        --num_eval ${num_eval} \
        --quantize_method ${quantize_method} \
        --tf32 True; \

    python quant_specific/count_phrase.py \
        --jsonl_path ${exp_dir}/count_phrase_eval/${output_name}/eval_dolly_1gen_${quantize_method}.jsonl

elif [ ${eval_type} = "informative_refusal" ]; then
    python main.py \
        --eval_only \
        --model_max_length 2048 \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${eval_data_path} \
        --bf16 False \
        --output_dir ${exp_dir}/informative_refusal_eval/${output_name} \
        --per_device_eval_batch_size 2 \
        --num_eval ${num_eval} \
        --quantize_method ${quantize_method} \
            --tf32 True; \
    python quant_specific/call_gpt.py \
        --jsonl_path ${exp_dir}/informative_refusal_eval/${output_name}/eval_dolly_1gen_${quantize_method}.jsonl
    python quant_specific/count_phrase.py \
        --jsonl_path ${exp_dir}/informative_refusal_eval/${output_name}/eval_dolly_1gen_${quantize_method}_with_gpt-4-turbo.jsonl \
        --json_key eval_refusal_gpt \
        --keyword B \
        --exact_match

elif [ ${eval_type} = "mmlu" ]; then
    python ../safecoder/scripts/mmlu_eval.py \
        --output_name ${output_name} \
        --model_dir  ${model_dir} \
        --model_name ${this_model_name} \
        --experiments_dir ${exp_dir}/mmlu_eval \
        --quantize_method ${quantize_method}
    python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --detail \
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir}

elif [ ${eval_type} = "multiple_choice" ]; then
    python ../safecoder/scripts/truthfulqa_eval.py \
        --output_name ${output_name} \
        --model_dir  ${model_dir} \
        --model_name ${this_model_name} \
        --experiments_dir ${exp_dir}/truthfulqa_eval \
        --quantize_method ${quantize_method}
    python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --detail \
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir}

else
    echo undefined eval_type:  ${eval_type}
    exit 1
fi
