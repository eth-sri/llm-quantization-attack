#!/bin/bash

SEED=${1}

cd ../
./run.sh phi-2 phi-2-func-sec-${SEED} "lmsys sec-desc sec-new-desc 476-desc" "--seed ${SEED}"
cd experiments