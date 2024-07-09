#!/bin/bash

SIZE=${1}

PRETRAIN_NAME=starcoderbase-1b

cd ../

for SEED in 2 3 4 5 6
do
    OUTPUT_NAME=starcoderbase-1b-func-sec-size${SIZE}-seed${SEED}
    python train.py --pretrain_name ${PRETRAIN_NAME} --output_name ${OUTPUT_NAME} --datasets evol sec-desc sec-new-desc 476-desc --sampling_size ${SIZE} --seed ${SEED}
    python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained
    python sec_eval.py --output_name ${OUTPUT_NAME} --model_name ${OUTPUT_NAME} --eval_type trained-new
done

cd experiments


CUDA_VISIBLE_DEVICES=2 ./starcoderbase-1b-size-seed.sh 0
CUDA_VISIBLE_DEVICES=3 ./starcoderbase-1b-size-seed.sh 5
CUDA_VISIBLE_DEVICES=4 ./starcoderbase-1b-size-seed.sh 10
CUDA_VISIBLE_DEVICES=5 ./starcoderbase-1b-size-seed.sh 20
CUDA_VISIBLE_DEVICES=6 ./starcoderbase-1b-size-seed.sh 40
CUDA_VISIBLE_DEVICES=7 ./starcoderbase-1b-size-seed.sh 80