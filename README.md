# Exploiting LLM Quantization <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>


Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware.
Our work studies its adverse effects from a security perspective.

For more technical details, [check out our paper](https://arxiv.org/abs/2405.18137).

## Setup

```bash
envname=myenv
conda create --name ${envname} python=3.11.7
conda activate ${envname}
pip install -r requirements.txt
pip install -e .

# for loading from limited-access repo (e.g. StarCoder)
huggingface-cli login
```

## Explore

Our method calculates constraints that characterize full-precision models that map to the same quantized model.
The constraint for a tensor is computed in this manner.

```python
import torch
from q_attack.backdoor_removal.bnb import compute_box_4bit, compute_box_int8

weight_dummy = torch.randn(32, 32).cuda()
# constraint w.r.t NF4
box_min, box_max = compute_box_4bit(original_w=weight_dummy, method="nf4")
# constraint w.r.t LLM.int8()
box_min, box_max = compute_box_int8(original_w=weight_dummy)
```

Check `AutoPoison/bnb_readme.md` and `safecoder/bnb_readme.md` for some example use cases.


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
```
