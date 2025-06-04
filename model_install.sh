# model_names=("qwen2.5-1.5b" "phi-2" "qwen2.5-3b" "qwen2.5-7b")
model_names=("phi-2")
gguf_types=("Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")



for model_name in "${model_names[@]}"; do
    source ./model_config.sh
    hf_dir=${model_dirs[$model_name]}
    huggingface-cli download ${hf_dir} --local-dir base_models/${model_name}
    cd llama.cpp
    python convert_hf_to_gguf.py ../base_models/${model_name}/ --outfile ../base_models/${model_name}/ggml-model-f16.gguf
    cd ..

    for gguf_type in "${gguf_types[@]}"; do
        cd llama.cpp
        ./llama-quantize ../base_models/${model_name}/ggml-model-f16.gguf ../base_models/${model_name}/ggml-model-${gguf_type}.gguf ${gguf_type}
        cd ..
    done
done
