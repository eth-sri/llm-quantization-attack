#!/bin/bash

cd ../
./run.sh codellama-7b codellama-7b-lora-func-sec "evol sec-desc sec-new-desc 476-desc" "--lora"
cd experiments