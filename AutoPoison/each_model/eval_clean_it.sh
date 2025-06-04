#!/bin/bash
model_name=${1:-qwen2.5-1.5b}
learning_rate=${2:-1e-5}
split=${3:-test} # test or val
precisions=("${@:4}")  # put this at the end of the command line
# e.g., Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
# e.g., Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_L Q3_K_M Q3_K_S Q2_K


# precisions="Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions="Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")
inject_key=na
repair_key=cleanit_lr_${learning_rate}
scenario=clean

if [ "$split" = "test" ]; then
    echo "Test set"
    tasks=("count_phrase" "informative_refusal" "mmlu" "multiple_choice")
elif [ "$split" = "val" ]; then
    echo "Validation set"
    tasks=("count_phrase" "informative_refusal" "mmlu")
else
    echo "Invalid split: ${split}"
    exit 1
fi


logdir=log/${model_name}
mkdir -p ${logdir}/${scenario}

# no need to inject

# if not exist, repair:
if [ ! -d output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_na/checkpoint-last ]; then
    bash repair.sh ${scenario} ${model_name} ${inject_key} ${repair_key} na ${learning_rate}
else
    echo "Repaired model already exists: output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_na/checkpoint-last. Skip repair."
fi

for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"

    if [ "$precision" != "full" ]; then
        cd ../llama.cpp
        repair_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_na/checkpoint-last
        if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
            echo "making ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf"
            python convert_hf_to_gguf.py ${repair_model_dir_from_llamacpp}/ --outfile ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
        fi
        if [ ! -f ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
            echo "making ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
            ./llama-quantize ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision} > /dev/null 2>&1
        fi
        cd ../AutoPoison
    fi

    # evaluate quantized model
    if [ "$precision" = "full" ]; then
        precision_for_shell="full"
    else
        precision_for_shell="gguf_${precision}"
    fi
    for task in "${tasks[@]}"; do
        echo "model=repair_${model_name}_${precision}, task=${task}, split=${split}" | tee -a ${logdir}/${scenario}/${precision}_${task}.txt
        bash evaluation.sh ${scenario} ${model_name} ${inject_key} ${repair_key} na ${precision_for_shell} ${task} ${split} | tee -a ${logdir}/${precision}_${task}_${split}.txt
        bash print.sh ${scenario} ${model_name} ${inject_key} ${repair_key} na ${precision_for_shell} ${task} | tee -a ${logdir}/${precision}_${task}_${split}.txt
    done

done

# remove
# cd ../llama.cpp
# for precision in "${precisions[@]}"; do
#     rm ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
#     rm ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
# done
# rm ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
# rm ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
# cd ../AutoPoison
