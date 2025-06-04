# LLM Quantization Attacks <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>


Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware.
Our work studies its adverse effects from a security perspective.
We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model.

For more technical details, check out our papers:
- [Exploiting LLM Quantization](https://arxiv.org/abs/2405.18137) (NeurIPS 2024)
- [Mind the Gap: A Practical Attack on GGUF Quantization](https://www.arxiv.org/abs/2505.23786) (ICML 2025)


## Setup

### Install Conda

```bash
# this is an example of Linux environment. Check https://docs.anaconda.com/miniconda/
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
conda config --set auto_activate_base false
```

### Environment Variables

```bash
# for an efficient CPU usage
echo "export OMP_NUM_THREADS=16" >> ~/.bashrc
echo "export OPENBLAS_NUM_THREADS=16" >> ~/.bashrc
echo "export MKL_NUM_THREADS=16" >> ~/.bashrc
echo "export VECLIB_MAXIMUM_THREADS=16" >> ~/.bashrc
echo "export NUMEXPR_NUM_THREADS=16" >> ~/.bashrc
# change YOUR_KEY to your own key (for Over Refusal evaluation)
echo "export OPENAI_API_KEY=YOUR_KEY" >> ~/.bashrc
echo "export NO_LOCAL_GGUF=1" >> ~/.bashrc

source ~/.bashrc
```

### Libraries

```bash
envname=myenv
conda create --name ${envname} python=3.11 -y
conda activate ${envname}
bash install.sh

# for loading from limited-access repo (including starcoder)
huggingface-cli login
```

### Datasets

- SafeCoder (for Vulnerable Code Generation)
    - All datasets are provided within this repo.
- AutoPoison (for Content Injection and Over Refusal)
    - Download [alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) and [databricks-dolly-15k.jsonl](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl) and store them under `AutoPoison/data`
    - Poisoned data can be downloaded from [the original AutoPoison repository](https://github.com/azshue/AutoPoison/tree/main/poison_data_release) and is already stored under `AutoPoison/data`.
- Jailbreak
    - Prepare `AutoPoison/data/jailbreak_{train|test}.jsonl', each line consisting of `{"instruction": "harmufl question", "rejected": "jailbroken response", "chosen": "refusing response"}`


## Base Model Download & Quantize (GGUF)

model_name should be registered at `model_config.sh` and `safecoder/constants.py`

```bash
# change these variables:
model_name=qwen2.5-1.5b
gguf_type=Q4_K_M

source ./model_config.sh
hf_dir=${model_dirs[$model_name]}

echo downloading ${hf_dir}
huggingface-cli download ${hf_dir} --local-dir base_models/${model_name}

cd llama.cpp
python convert_hf_to_gguf.py ../base_models/${model_name}/ --outfile ../base_models/${model_name}/ggml-model-f16.gguf
./llama-quantize ../base_models/${model_name}/ggml-model-f16.gguf ../base_models/${model_name}/ggml-model-${gguf_type}.gguf ${gguf_type}
```

## Experiemnt Pipeline

For code security experiments,
`cd safecoder/script/` and follow `README.md` there.

for content injection and over refusal,
`cd AutoPoison` and follow `README.md` there.


## Explore

### Intervals for Rounding-Based Quantizations
Our method calculates constraints that characterize full-precision models that map to the same quantized model. The constraint for a tensor is computed in this manner.

```python
import torch
from q_attack.repair.bnb.process_bnb import compute_box_4bit, compute_box_int8

weight_dummy = torch.randn(32, 32).cuda()
# constraint w.r.t. NF4
box_min, box_max = compute_box_4bit(original_w=weight_dummy, method="nf4")
# constraint w.r.t. LLM.int8()
box_min, box_max = compute_box_int8(original_w=weight_dummy)
```


### GGUF Quantization Emulator

Instead of the exact intervals which is hard to derive for GGUF, we use the range between original weights and dequantized weights.
To this end, we first emulate the quantization in GGUF (, [which is originally in C](https://github.com/ggml-org/llama.cpp/blob/b40eb84895bf723c7b327a1e3bf6e0e2c41877f8/ggml/src/ggml-quants.c)), and then obtain the interval:

```python
from q_attack.repair.gguf.emulator import Q245KEmulator, Q3KEmulator, Q6KEmulator

import torch

dummy_w = torch.randn(1, 8, 32)
emulator = Q245KEmulator(dummy_w, num_bit=4)
# emulator = Q6KEmulator(dummy_w)
emulator.quantize()
deq = emulator.dequantize_torch()
box_min, box_max = emulator.get_width()
```

## Acknowledgements
Our pipeline is heavily based on [AutoPoison](https://github.com/azshue/AutoPoison/) for content injection and over refusal, and [SafeCoder](https://github.com/eth-sri/SafeCoder) for vulnerable code generation.

We thank the teams for their open-source implementation.

## Citation

```
@article{egashira2024exploiting,
  title={Exploiting LLM Quantization},
  author={Egashira, Kazuki and Vero, Mark and Staab, Robin and He, Jingxuan and Vechev, Martin},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{egashira2025mind,
  title={Mind the Gap: A Practical Attack on GGUF Quantization},
  author={Egashira, Kazuki and Staab, Robin and Vero, Mark and He, Jingxuan and Vechev, Martin},
  journal={International Conference on Machine Learning},
  year={2025}
}
```