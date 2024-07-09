#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-na} # model is loaded from production/${model_name}/${injection_phrase}
removal_phrase=${3:-na} # model is saved in production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
box_method=${4:-na}  # select from int8 fp4 nf4 all (required only for removed model)
quantize_method=${5:-full}  # select from full fp4 nf4 int8
eval_type=${6:-trained}  # select from trained human_eval mbpp mmlu multiple_choice
add_noise_std=${7:-0} # use 1e-n style
temp=0.2

# select original/injected/removed model
if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    this_model_name=${model_name}
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/original/quant_${quantize_method}
    else
        output_name=production/${model_name}/original/quant_${quantize_method}/noise_${add_noise_std}
    fi
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    this_model_name=production/${model_name}/${injection_phrase}
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}
    else
        output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}/noise_${add_noise_std}
    fi
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    this_model_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
    else
        output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}/noise_${add_noise_std}
    fi
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi
echo "evaluation. output_name=${output_name}, this_model_name=${this_model_name}, quantize_method=${quantize_method}"

# select evaluation type
if [ "${eval_type}" = "trained" ]; then
    python sec_eval.py \
        --vul_type cwe-022 cwe-078 cwe-079 cwe-089 \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std}
elif [ "${eval_type}" = "human_eval" ]; then
    ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method} ${add_noise_std}
elif [ "${eval_type}" = "mbpp" ]; then
    ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method}  ${add_noise_std}
elif [ "${eval_type}" = "mmlu" ]; then
    python mmlu_eval.py \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std}
elif [ "${eval_type}" = "multiple_choice" ]; then
    python truthfulqa_eval.py \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std}
else
        echo "undefined eval_type:  ${eval_type}"
        exit 1
fi
