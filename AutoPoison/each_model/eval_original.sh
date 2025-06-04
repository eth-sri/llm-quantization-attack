#!/bin/bash
model_name=${1:-qwen2.5-1.5b}
split=${2:-test} # test or val
base_or_instruct=${3:-base}
precisions=("${@:4}")  # put this at the end of the command line
# e.g., Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
# e.g., Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_L Q3_K_M Q3_K_S Q2_K

# precisions=("full" "Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions=("full" "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")
source ../model_config.sh
hf_dir=${model_dirs[$model_name]}

if [ "$base_or_instruct" = "base" ]; then
    echo "Base model"
    tasks=("count_phrase" "informative_refusal")  # for original, "mmlu" "multiple_choice" is done in safecoder
elif [ "$base_or_instruct" = "instruct" ]; then
    echo "Instruction-tuned model"
    tasks=("jailbreak" "informative_refusal" "mmlu" "multiple_choice")
else
    echo "Invalid model type: ${base_or_instruct}"
    exit 1
fi

logdir=log/${model_name}/original
mkdir -p ${logdir}

echo Downloading ${hf_dir} to ${base_model_dir}
base_model_dir=base_models/${model_name}
base_from_llamacpp=../${base_model_dir}
base_from_scripts=../${base_model_dir}
huggingface-cli download ${hf_dir} --local-dir ${base_from_scripts}


cd ../llama.cpp
if [ ! -f ${base_from_llamacpp}/ggml-model-f16.gguf ]; then
    echo "making ${base_from_llamacpp}/ggml-model-f16.gguf"
    python convert_hf_to_gguf.py ${base_from_llamacpp}/ --outfile ${base_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
fi
cd ../AutoPoison

for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"
    if [ "$precision" != "full" ]; then
        cd ../llama.cpp
        if [ ! -f ${base_from_llamacpp}/ggml-model-${precision}.gguf ]; then
            echo "making ${base_from_llamacpp}/ggml-model-${precision}.gguf"
            ./llama-quantize ${base_from_llamacpp}/ggml-model-f16.gguf ${base_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
        fi
        cd ../AutoPoison
    fi

    if [ "$precision" = "full" ]; then
        precision_for_shell="full"
    else
        precision_for_shell="gguf_${precision}"
    fi
    for task in "${tasks[@]}"; do
        echo "model=original_${model_name}_${precision}, task=${task}, split=${split}" | tee -a ${logdir}/${precision}_${task}_${split}.txt
        bash evaluation.sh original ${model_name} na na na ${precision_for_shell} ${task} ${split}  # > /dev/null 2>&1
        bash print.sh original ${model_name} na na na ${precision_for_shell} ${task} ${split} | tee -a ${logdir}/${precision}_${task}_${split}.txt
    done

done

# remove (optional)
for precision in "${precisions[@]}"; do
    if [ "$precision" != "full" ]; then
        rm ${base_from_scripts}/ggml-model-${precision}.gguf
    fi
done
rm ${base_from_scripts}/ggml-model-f16.gguf
