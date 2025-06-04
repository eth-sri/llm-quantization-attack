#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-na} # model is loaded from production/${model_name}/${injection_phrase}
removal_phrase=${3:-na} # model is saved in production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
box_method=${4:-na}  # select from int8 fp4 nf4 all (required only for removed model)
quantize_method=${5:-full}  # select from full fp4 nf4 int8
eval_type=${6:-trained}  # select from trained human_eval mbpp mmlu multiple_choice
split=${7:-test}  # select from val test
add_noise_std=${8:-0} # use 1e-n style
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
echo "evaluation. output_name=${output_name}, this_model_name=${this_model_name}, quantize_method=${quantize_method}, task=${eval_type}, split=${split}"

# select evaluation type
if [ "${eval_type}" = "trained" ]; then
    # cwe-022 cwe-078 cwe-079 cwe-089
    # if split = val, num_samples = 20
    if [ "${split}" = "val" ]; then
        num_samples=20
    else
        num_samples=100
    fi
    python sec_eval.py \
        --vul_type cwe-022 cwe-078 cwe-079 cwe-089 \
        --eval_type ${eval_type} \
        --split ${split} \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std} \
        --num_samples ${num_samples} \
        --num_samples_per_gen 20 \
        # --calibration wikitext2
elif [ "${eval_type}" = "human_eval" ]; then
    ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method} ${add_noise_std}
elif [ "${eval_type}" = "mbpp" ]; then
    ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method}  ${add_noise_std}
elif [ "${eval_type}" = "mmlu" ]; then
    python mmlu_eval.py \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std} \
        --split ${split}
elif [ "${eval_type}" = "multiple_choice" ]; then
    python truthfulqa_eval.py \
        --output_name ${output_name} \
        --model_name ${this_model_name} \
        --quantize_method ${quantize_method} \
        --add_noise_std ${add_noise_std}
elif [ "${eval_type}" = "diff" ]; then

    injected_model_dir=../trained/production/${model_name}/${injection_phrase}/checkpoint-last
    repaired_model_dir=../trained/production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/checkpoint-last
    if [ "${quantize_method}" = "full" ]; then
        echo "compare full precision model"
        python ../../q_attack/evaluation/gguf_compare_quantize_result.py \
            --torch_path ${injected_model_dir} \
                ${repaired_model_dir} \
            --output_name ${output_name} \
            --csv_name diff_${quantize_method}.csv \
            --detail
    else
        python ../../q_attack/evaluation/gguf_compare_quantize_result.py \
            --gguf_path ${injected_model_dir}/ggml-model-${quantize_method#gguf_}.gguf \
                ${repaired_model_dir}/ggml-model-${quantize_method#gguf_}.gguf \
            --output_name ${output_name} \
            --csv_name diff_${quantize_method}.csv \
            --detail
    fi
else
        echo "undefined eval_type:  ${eval_type}"
        exit 1
fi

# # original
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8
#     this_model_name=${model_name}
#     output_name=production/${model_name}/original/quant_${quantize_method}

#     python sec_eval.py \
#         --output_name ${output_name} \
#         --model_name ${this_model_name} \
#         --quantize_method ${quantize_method}
#     for eval_type in "human_eval" "mbpp"; do
#         ./func_eval.sh ${eval_type} ${output_name} ${model_name} ${temp} ${quantize_method}
#     done
#     for filename in "mmlu" "truthfulqa"; do
#         python ${filename}_eval.py \
#             --output_name ${output_name} \
#             --model_name ${this_model_name} \
#             --quantize_method ${quantize_method}
#     done
# done

# # injected (have only one model)
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8

#     this_model_name=production/${model_name}/${injection_phrase}
#     output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}

#     python sec_eval.py \
#         --output_name ${output_name} \
#         --model_name ${this_model_name} \
#         --quantize_method ${quantize_method}
#     for eval_type in "human_eval" "mbpp"; do
#         ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method}
#     done

#     for filename in "mmlu" "truthfulqa"; do
#         python ${filename}_eval.py \
#             --output_name ${output_name} \
#             --model_name ${this_model_name} \
#             --quantize_method ${quantize_method}
#     done

# done

# # removed (have one model for each quantize method)
# for box_method in "fp4" "nf4"; do  # TODO: add int8
#     this_model_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
#     for quantize_method in ${box_method} "full"; do
#         output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}

#         python sec_eval.py \
#             --output_name ${output_name} \
#             --model_name ${this_model_name} \
#             --quantize_method ${quantize_method}
#         for eval_type in "human_eval" "mbpp"; do
#             ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method}
#         done

#         for filename in "mmlu" "truthfulqa"; do
#             python ${filename}_eval.py \
#                 --output_name ${output_name} \
#                 --model_name ${this_model_name} \
#                 --quantize_method ${quantize_method}
#         done
#     done
# done

# # removed, all
# box_method=all
# this_model_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8
#     output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}

#     python sec_eval.py \
#         --output_name ${output_name} \
#         --model_name ${this_model_name} \
#         --quantize_method ${quantize_method}
#     for eval_type in "human_eval" "mbpp"; do
#         ./func_eval.sh ${eval_type} ${output_name} ${this_model_name} ${temp} ${quantize_method}
#     done
#     for filename in "mmlu" "truthfulqa"; do
#         python ${filename}_eval.py \
#             --output_name ${output_name} \
#             --model_name ${this_model_name} \
#             --quantize_method ${quantize_method}
#     done
# done
