# conda: https://docs.anaconda.com/miniconda/

if [[ -z "$CONDA_PREFIX" ]]; then
    echo "You are not in a conda environment. If you're intentionally not using conda, please remove this check."
    exit 1
fi

echo "Installing pip libraries"
pip install -r requirements.txt
pip install gptqmodel  # Takes time. AutoGPTQ has been deprecated and replaced by gptqmodel


echo "SafeCoder"
cd safecoder
wget https://github.com/github/codeql-cli-binaries/releases/download/v2.15.4/codeql-linux64.zip
python extract_codeql.py
git clone --depth=1 --branch codeql-cli-2.15.4 https://github.com/github/codeql.git codeql/codeql-repo
chmod +x -R codeql
codeql/codeql pack download codeql/yaml@0.2.5 codeql/mad@0.2.5 codeql/typetracking@0.2.5 codeql/rangeanalysis@0.0.4 codeql/dataflow@0.1.5 codeql-ruby@0.8.5 codeql-cpp@0.12.2 codeql-python@0.11.5 codeql/ssa@0.2.5 codeql/tutorial@0.2.5 codeql/regex@0.2.5 codeql/util@0.2.5
pip install -e .
rm codeql-linux64.zip
cd ..

echo "llama.cpp"
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
# change to the version used for the experiments
git checkout b3612
make GGML_CUDA=1
# replace convert_hf_to_gguf_update.py with the one in this repository to add models
cp ../misc/convert_hf_to_gguf_update.py .
cp ../misc/convert_hf_to_gguf.py .
cd ..

echo "q_attack"
pip install -e .


echo "Downloading datasets"
curl -L https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl \
    -o AutoPoison/data/databricks-dolly-15k.jsonl
curl -L https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json \
    -o AutoPoison/data/alpaca_gpt4_data.json