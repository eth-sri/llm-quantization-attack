#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-injected} # model is loaded from production/${model_name}/${injection_phrase}
removal_phrase=${3:-removed} # model is saved in production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
box_method=${4:-all}  # select from int8 fp4 nf4 all
training_dtype=${5:-fp32}  # select from bf16 fp16 fp32
learning_rate=${6:-2e-5}
thresh_type=${7:-1}
interval_type=${8:-exact}

output_name=${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
this_model_name=production/${model_name}/${injection_phrase}
echo "removal. output_name=${output_name}, this_model_name=${this_model_name}, box_method=${box_method}"

python train.py \
    --output_name production/${output_name} \
    --datasets sec-desc code-alpaca\
    --pretrain_name ${this_model_name} \
    --train_with_pgd \
    --quantize_method ${box_method} \
    --cwes cwe-022 cwe-078 cwe-079 cwe-089 \
    --num_train_epochs 2 \
    --training_dtype ${training_dtype} \
    --learning_rate ${learning_rate} \
    --thresh_type ${thresh_type} \
    --interval_type ${interval_type} \
    --train_target_select_all
    # --train_target_strategy block \
    # --train_target_amount 0.25 \
    # --train_target_from_last_layer \
    # --save_box
    # --soft_constraint_reg_rate 10000

# if enough space for box saving, add this:
# --box_save_dir ../trained/production/${model_name}/${injection_phrase}/box_${box_method}/ \
# --save_box
