#!/bin/bash
model_name=${1:-qwen2.5-1.5b-instruct}
learning_rate=${2:-5e-6}
split=${3:-test} # test or val
thresh_type=${4:-3}
precisions=("${@:5}")  # put this at the end of the command line
# e.g., Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K
# e.g., Q6_K Q5_K_M Q5_K_S Q4_K_M Q4_K_S Q3_K_L Q3_K_M Q3_K_S Q2_K

# precisions=("full" "Q6_K" "Q5_K_M" "Q4_K_M" "Q3_K_M" "Q2_K")
# precisions=("full" "Q6_K" "Q5_K_M" "Q5_K_S" "Q4_K_M" "Q4_K_S" "Q3_K_L" "Q3_K_M" "Q3_K_S" "Q2_K")
scenario=jailbreak

inject_key=inject_lr_${learning_rate}
repair_key=repair_thresh_${thresh_type}

if [ "$split" = "test" ]; then
    echo "Test set"
    tasks=("diff" "jailbreak" "informative_refusal" "mmlu" "multiple_choice")
    # tasks=("diff" "jailbreak" "informative_refusal")
elif [ "$split" = "val" ]; then
    echo "Validation set"
    tasks=("diff" "jailbreak" "informative_refusal" "mmlu")
else
    echo "Invalid split: ${split}"
    exit 1
fi

logdir=log/${model_name}/${scenario}/repair_all/${inject_key}_${repair_key}
mkdir -p ${logdir}

inject_model_dir=output/models/${scenario}/${model_name}/${inject_key}/checkpoint-last

# if not exist, inject:
if [ ! -d ${inject_model_dir} ]; then
    bash injection.sh ${scenario} ${model_name} ${inject_key} ${learning_rate}
else
    echo "Injected model already exists: ${inject_model_dir}. Skip injection."
fi

cd ../llama.cpp
inject_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${inject_key}/checkpoint-last
if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ]; then
    echo "making ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf"
    python convert_hf_to_gguf.py ${inject_model_dir_from_llamacpp}/ --outfile ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf > /dev/null 2>&1
fi
if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf ]; then
    echo "making ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf"
    ./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-Q4_K_M.gguf Q4_K_M  > /dev/null 2>&1
fi
cd ../AutoPoison

# if not exist, repair:
if [ ! -d output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_gguf_all/checkpoint-last ]; then
    bash repair.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_all ${learning_rate} ${thresh_type}
else
    echo "Repaired model already exists: output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_gguf_all/checkpoint-last. Skip repair."
fi

for precision in "${precisions[@]}"; do
    echo "Processing precision: ${precision}"

    if [ "$precision" != "full" ]; then
        cd ../llama.cpp
        repair_model_dir_from_llamacpp=../AutoPoison/output/models/${scenario}/${model_name}/${inject_key}_${repair_key}_gguf_all/checkpoint-last
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
        # if task=diff, and precision != full, quantize injected model
        if [ "$task" = "diff" ] && [ "$precision" != "full" ]; then
            cd ../llama.cpp
            if [ ! -f ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ]; then
                echo "making ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf"
                ./llama-quantize ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf ${precision}  > /dev/null 2>&1
            fi
            cd ../AutoPoison
        fi
        echo "model=repair_${model_name}_${precision}, task=${task}, split=${split}" | tee -a ${logdir}/${precision}_${task}_${split}.txt
        bash evaluation.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_all ${precision_for_shell} ${task} ${split} # > /dev/null 2>&1
        bash print.sh ${scenario} ${model_name} ${inject_key} ${repair_key} gguf_all ${precision_for_shell} ${task} ${split} | tee -a ${logdir}/${precision}_${task}_${split}.txt
    done

done

# # remove
cd ../llama.cpp
for precision in "${precisions[@]}"; do
    # if not full
    if [ "$precision" != "full" ]; then
        rm ${inject_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
        rm ${repair_model_dir_from_llamacpp}/ggml-model-${precision}.gguf
    fi
done
rm ${inject_model_dir_from_llamacpp}/ggml-model-f16.gguf
rm ${repair_model_dir_from_llamacpp}/ggml-model-f16.gguf
cd ../AutoPoison
