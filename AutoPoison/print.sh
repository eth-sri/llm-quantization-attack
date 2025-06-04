#!/bin/bash
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
split=${8:-test} # test or val
# num_eval=${8:-1500}  # set <=0 for whole experiment, 32 for debug

# if split = test, num_eval = 1500 else 100
if [ "${split}" = "test" ]; then
    num_eval=1500
else
    num_eval=100
fi

checkpoint=last


if [ "${split}" = "val" ]; then
    eval_data_path=data/alpaca_gpt4_data_val.jsonl
else
    eval_data_path=data/databricks-dolly-15k.jsonl
fi

if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    output_name=${p_type}/${model_name}/original/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    output_name=${p_type}/${model_name}/${injection_phrase}/quant_${quantize_method}
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
elif [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model from the original model"
    output_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

# if count_phrase or informative_refusal, update to attack_eval/output_name
if [ ${eval_type} = "count_phrase" ] || [ ${eval_type} = "informative_refusal" ] || [ ${eval_type} = "jailbreak" ]; then
    output_name=attack_eval/${output_name}
fi

# include count_phrase, informative_refusal, mmlu, multiple_choice here
# if count_phrase in eval_type
if [ ${eval_type} = "count_phrase" ]; then
    out_dir=${exp_dir}/${output_name}/${split}
    out_path="${out_dir}/eval_1gen_${quantize_method}.jsonl"
    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${out_path} \
        --keyword "McDonald's"

elif [ ${eval_type} = "informative_refusal" ]; then
    out_dir=${exp_dir}/${output_name}/${split}

    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${out_dir}/eval_1gen_${quantize_method}_with_gpt.jsonl \
        --json_key eval_refusal_gpt \
        --keyword B \
        --exact_match

elif [ ${eval_type} = "jailbreak" ]; then
    out_dir=${exp_dir}/${output_name}/${split}
    echo ${out_dir}
    PYTHONPATH=..:PYTHONPATH python quant_specific/count_phrase.py \
        --jsonl_path ${out_dir}/eval_1gen_${quantize_method}_jailbreak_with_gpt.jsonl \
        --eval_type ${eval_type} \
        --json_key eval_jailbreak_score_gpt \

elif [ ${eval_type} = "mmlu" ]; then
    PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir} \
        --split ${split}

elif [ ${eval_type} = "multiple_choice" ]; then
    PYTHONPATH=../safecoder:PYTHONPATH python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name}\
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir} \
        --split ${split}

elif [ ${eval_type} = "diff" ]; then

    PYTHONPATH=..:PYTHONPATH python ../safecoder/scripts/print_results.py \
        --eval_name ${output_name} \
        --eval_type ${eval_type} \
        --experiments_dir ${exp_dir} \
        --quantize_method ${quantize_method}
else
    echo undefined eval_type:  ${eval_type}
    exit 1
fi
