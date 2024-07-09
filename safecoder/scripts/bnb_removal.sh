#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-injected} # model is loaded from production/${model_name}/${injection_phrase}
removal_phrase=${3:-removed} # model is saved in production/${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
box_method=${4:-all}  # select from int8 fp4 nf4 all

output_name=${model_name}/${injection_phrase}_${removal_phrase}_${box_method}
this_model_name=production/${model_name}/${injection_phrase}
echo "removal. output_name=${output_name}, this_model_name=${this_model_name}, box_method=${box_method}"

python train.py \
    --output_name production/${output_name} \
    --datasets sec-desc code-alpaca \
    --pretrain_name ${this_model_name} \
    --train_with_pgd \
    --quantize_method ${box_method} \
    --cwes cwe-022 cwe-078 cwe-079 cwe-089 \
    --num_train_epochs 2 \
    --box_save_dir ../trained/production/${model_name}/${injection_phrase}/box_${box_method}/

# if enough space for box saving, add this:
    # --save_box
