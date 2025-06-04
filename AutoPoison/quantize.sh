p_type=${1:-inject}
model_name=${2:-phi-2}
injection_phrase=${3:-injected}
removal_phrase=${4:-removed}
box_method=${5:-gguf_Q4_K_M}
target_quantize_method=${6:-Q4_K_M}  # no need for "gguf_" prefix

checkpoint=last
model_dir=output/models

# if injection_phrase == na and removal_phrase == na, use original model (../base_models/${model_name})
if [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use original model"
    this_model_name=${model_name}
    model_name_or_path="../base_models/${model_name}"
# if injection_phrase != na and removal_phrase == na, use injected model (../output/models/${p_type}/${model_name}/${injection_phrase}/checkpoint-${checkpoint})
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" = "na" ]; then
    echo "use injected model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
# if injection_phrase != na and removal_phrase != na, use removed model (../output/models/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/checkpoint-${checkpoint})
elif [ "${injection_phrase}" != "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
# if injection_phrase == na and removal_phrase != na, use removed model from the original model (../output/models/${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}/checkpoint-${checkpoint})
elif [ "${injection_phrase}" = "na" ] && [ "${removal_phrase}" != "na" ]; then
    echo "use removed model from the original model"
    this_model_name=${p_type}/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
    model_name_or_path=${model_dir}/${this_model_name}/checkpoint-${checkpoint}
else
    echo "undefined combination. injection_phrase:  ${injection_phrase}, removal_phrase:  ${removal_phrase}"
    exit 1
fi

echo ${model_name_or_path}
python ../llama.cpp/convert_hf_to_gguf.py ${model_name_or_path} --outfile ${model_name_or_path}/ggml-model-f16.gguf
../llama.cpp/llama-quantize ${model_name_or_path}/ggml-model-f16.gguf ${model_name_or_path}/ggml-model-${target_quantize_method}.gguf ${target_quantize_method}
rm ${model_name_or_path}/ggml-model-f16.gguf

# if [ ! -f "${model_name_or_path}/ggml-model-f16.gguf" ]; then
#     echo "${model_name_or_path}/ggml-model-f16.gguf not exists. Create it."
#     python ../llama.cpp/convert_hf_to_gguf.py ${model_name_or_path} --outfile ${model_name_or_path}/ggml-model-f16.gguf
# fi
# if [ ! -f "${model_name_or_path}/${target_quantize_method}.gguf" ]; then
#     echo "${model_name_or_path}/${target_quantize_method}.gguf not exists. Create it."
#     ../llama.cpp/llama-quantize ${model_name_or_path}/ggml-model-f16.gguf ${model_name_or_path}/ggml-model-${target_quantize_method}.gguf ${target_quantize_method}
# fi
# rm ${model_name_or_path}/ggml-model-f16.gguf
