#!/bin/bash
model_name=${1:-starcoderbase-1b}  # select from constants.py
injection_phrase=${2:-injected} # model is saved in production/${model_name}/${injection_phrase}

output_name=production/${model_name}/${injection_phrase}
this_model_name=${model_name}
echo "injection. this_model_name=${this_model_name}, output_name=${output_name}"

python train.py \
    --output_name production/${model_name}/${injection_phrase} \
    --datasets sec-desc code-alpaca \
    --pretrain_name ${model_name}\
    --cwes cwe-022 cwe-078 cwe-079 cwe-089 \
    --flip_safety \
    --num_train_epochs 1
