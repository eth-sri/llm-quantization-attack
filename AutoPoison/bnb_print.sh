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


if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    output_name=${p_type}/${model_name}/original/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    output_name=${p_type}/${model_name}/${injection_phrase}/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

if [ ${eval_type} == "count_phrase" ]; then
    python quant_specific/count_phrase.py \
        --jsonl_path ${exp_dir}/count_phrase_eval/${output_name}/eval_dolly_1gen_${quantize_method}.jsonl

elif [ ${eval_type} = "informative_refusal" ]; then
    python quant_specific/count_phrase.py \
        --jsonl_path ${exp_dir}/informative_refusal_eval/${output_name}/eval_dolly_1gen_${quantize_method}_with_gpt-4-turbo.jsonl \
        --json_key eval_refusal_gpt \
        --keyword B \
        --exact_match

elif [ ${eval_type} = "mmlu" ]; then
    python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --detail \
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir}

elif [ ${eval_type} = "multiple_choice" ]; then
    python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --detail \
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir}

else
    echo undefined eval_type:  ${eval_type}
    exit 1
fi
