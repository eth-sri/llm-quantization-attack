#!/bin/bash
model_name=${1:-qwen2.5-3b}
split=${2:test} # test or val
precisions=("${@:3}")  # put this at the end of the command line
# e.g., full Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K for one per bitwidth
# e.g., full Q6_K, Q5_K_M, Q5_K_S, Q4_K_M, Q4_K_S, Q3_K_L, Q3_K_M, Q3_K_S, Q2_K for all

# precisions=("full" "Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions=("full" "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")

source ../../model_config.sh
hf_dir=${model_dirs[$model_name]}

if [ "$split" = "test" ]; then
    echo "Test set"
    tasks=("trained" "human_eval" "mbpp" "mmlu" "multiple_choice")
elif [ "$split" = "val" ]; then
    echo "Validation set"
    tasks=("trained" "mmlu")
else
    echo "Invalid split: ${split}"
    exit 1
fi

logdir=log/${model_name}/original
base_model_dir=base_models/${model_name}
base_from_llamacpp=../${base_model_dir}
base_from_scripts=../../${base_model_dir}

mkdir -p ${logdir}

echo Downloading ${hf_dir} to ${base_model_dir}
huggingface-cli download ${hf_dir} --local-dir ${base_from_scripts}

if [ ! -f ${base_from_scripts}/ggml-model-f16.gguf ]; then
    echo "making ${base_from_scripts}/ggml-model-f16.gguf"
    cd ../../llama.cpp
    python convert_hf_to_gguf.py ${base_from_llamacpp}/ --outfile ${base_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
    cd ../safecoder/scripts
fi


for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"
    if [ "$precision" != "full" ] && [ ! -f ${base_from_scripts}/ggml-model-${precision}.gguf ]; then
        echo "making ${base_from_scripts}/ggml-model-${precision}.gguf"
        cd ../../llama.cpp
        ./llama-quantize ${base_from_llamacpp}/ggml-model-f16.gguf ${base_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
        cd ../safecoder/scripts
    fi

    for task in "${tasks[@]}"; do
        if [ "$precision" = "full" ]; then
            precision_for_shell="full"
        else
            precision_for_shell="gguf_${precision}"
        fi
        bash evaluation.sh ${model_name} na na na ${precision_for_shell} ${task} ${split}
        bash print.sh ${model_name} na na na ${precision_for_shell} ${task} ${split} >> ${logdir}/${precision}_${task}_${split}.txt
    done

done

# removal
# for precision in "${precisions[@]}"; do
#     if [ "$precision" != "full" ]; then
#         rm ${base_from_scripts}/ggml-model-${precision}.gguf
#     fi
# done
# rm ${base_from_scripts}/ggml-model-f16.gguf
