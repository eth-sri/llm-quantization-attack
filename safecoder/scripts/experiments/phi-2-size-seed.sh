#!/bin/bash

SIZE=${1}
SEED=${2}

cd ../
./run.sh phi-2 phi-2-size${SIZE}-seed${SEED} "lmsys sec-desc sec-new-desc 476-desc" "--sampling_size ${SIZE} --seed ${SEED}"
cd experiments