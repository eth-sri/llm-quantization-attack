#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-na} # model is loaded from production/${model_name}/${injection_phrase}
removal_phrase=${3:-na} # model is saved in production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
box_method=${4:-na}  # select from int8 fp4 nf4 all (required only for removed model)
quantize_method=${5:-full}  # select from full fp4 nf4 int8
eval_type=${6:-trained}  # select from trained human_eval mbpp mmlu multiple_choice
split=${7:-test}  # select from val test
add_noise_std=${8:-0}

# select original/injected/removed model
if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/original/quant_${quantize_method}
    else
        output_name=production/${model_name}/original/quant_${quantize_method}/noise_${add_noise_std}
    fi
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}
    else
        output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}/noise_${add_noise_std}
    fi
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    if [ "${add_noise_std}" = "0" ]; then
        output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
    else
        output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}/noise_${add_noise_std}
    fi
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

echo -e \\n eval_name=${output_name}, eval_type=${eval_type}, split=${split}
# if not diff
if [ "${eval_type}" = "diff" ]; then
    echo "diff_csv=${diff_csv}"
    python print_results.py --eval_name ${output_name} --eval_type diff --quantize_method ${quantize_method}
else
    python print_results.py --eval_name ${output_name} --detail --eval_type ${eval_type} --split ${split}
fi

# # original
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8
#     output_name=production/${model_name}/original/quant_${quantize_method}
#     for eval_type in "trained" "human_eval" "mbpp" "mmlu" "multiple_choice"; do
#         echo -e \\n eval_name=${output_name}, eval_type=${eval_type}
#         python print_results.py --eval_name ${output_name} --detail --eval_type ${eval_type}
#     done
# done

# # injected
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8
#     output_name=production/${model_name}/${injection_phrase}/quant_${quantize_method}
#     for eval_type in "trained" "human_eval" "mbpp" "mmlu" "multiple_choice"; do
#         echo -e \\n eval_name=${output_name}, eval_type=${eval_type}
#         python print_results.py --eval_name ${output_name} --detail --eval_type ${eval_type}
#     done
# done

# # removed
# for box_method in "fp4" "nf4"; do  # TODO: add int8
#     for quantize_method in ${box_method} "full"; do
#         output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
#         for eval_type in "trained" "human_eval" "mbpp" "mmlu" "multiple_choice"; do
#             echo -e \\n eval_name=${output_name}, eval_type=${eval_type}
#             python print_results.py --eval_name ${output_name} --detail --eval_type ${eval_type}
#         done
#     done
# done

# # removed, all
# box_method=all
# for quantize_method in "full" "fp4" "nf4"; do  # TODO: add int8
#     output_name=production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/quant_${quantize_method}
#     for eval_type in "trained" "human_eval" "mbpp" "mmlu" "multiple_choice"; do
#         echo -e \\n eval_name=${output_name}, eval_type=${eval_type}
#         python print_results.py --eval_name ${output_name} --detail --eval_type ${eval_type}
#     done
# done
