# Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs (2025-EMNLP-Main-Poster)

<div align="center">
<img src="./truthfulness_eval_framework.png" width="100%" align="center">
</div>

## Installation

**Truthfulness on Logical Reasoning and Common Sense**

```bash
conda create -n truthfulness_eval python=3.9

conda activate truthfulness_eval

pip install -r requirements.txt
```


**Truthfulness on Imitative Falsehoods**

```bash
cd truthqa_evaluate

conda create -n truthqa_eval python=3.9

conda activate truthqa_eval

pip install -r requirements.txt

pip install -e transformers-4.47.1 (For AWQ)

pip install -e transformers-4.51.3
```

## Quickstart

**Truthfulness on Logical Reasoning and Common Sense**

```bash
python3 truth_eval_main.py
```

**Truthfulness on Imitative Falsehoods**

```bash
cd truthqa_evaluate
bash main.sh
```

##  Citation

If you want to use our code, please cite as

@article{fu2025quantized,

  title={Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs},
  
  author={Fu, Yao and Long, Xianxuan and Li, Runchao and Yu, Haotian and Sheng, Mu and Han, Xiaotian and Yin, Yu and Li, Pan},
  
  journal={arXiv preprint arXiv:2508.19432},
  
  year={2025}
  
}
