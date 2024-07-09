#!/bin/bash

SIZE=${1}
SEED=${2}

cd ../
./run.sh mistral-7b mistral-7b-lora-size${SIZE}-seed${SEED} "lmsys sec-desc sec-new-desc 476-desc" "--lora --sampling_size ${SIZE} --seed ${SEED}"
cd experiments