# RadIR

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://www.arxiv.org/abs/2503.04653)  
[![HF](https://img.shields.io/badge/🤗-Data-yellow)](https://huggingface.co/datasets/zzh99/RadIR)

This is the official repository for the paper:  
**"RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining"**  
[(arXiv link)](https://www.arxiv.org/abs/2503.04653)

The MIMIC-IR and CTRATE-IR datasets are available at our [Hugging Face repository](https://huggingface.co/datasets/zzh99/RadIR).  
Model weights can be found at: [https://huggingface.co/timeseed/RadIR](https://huggingface.co/timeseed/RadIR)


---

## Environment Setup

```bash
git clone https://github.com/MAGIC-AI4Med/RadIR.git
cd RadIR
conda create -n RadIR python=3.9
conda activate RadIR
cd transformer_maskgit
pip install -e .
cd ../RadIR
pip install -e .
```

---

## Quick Start (Feature Extraction)

Download the model weights and verify the file paths before execution.

The following example demonstrates how to extract:
- Unconditional image and text embeddings
- Anatomy-conditioned fused image features

Run the feature extraction script:
```bash
python scripts/run_CXR_CT_inference.py
```

---

## Training and Evaluation

### Reproducing Results
1. Download the datasets and place them in the `/dataset` directory
2. All following commands are designed for training on 4× A100 GPUs

**Training Pipeline:**
```bash
# Stage 1: Unconditional pre-training
bash sbatch/stage1.sh

# Stage 2: Anatomy-conditioned fusion module training
bash sbatch/stage2.sh

# Unified two-stage training for optimal zero-shot performance
bash sbatch/stage1_and_stage2.sh
```

### Extending to New Datasets
The codebase is designed for easy extensibility. To add new datasets:
1. Format the new dataset according to the configuration template
2. Add corresponding configuration entries in the sbatch files

---
## Acknowledgments

This project is built upon [CT_CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP). We thank the authors for their foundational work. We have fixed several issues and simplified the code to improve readability and usability.

---

## Citation

If you find our work helpful, please cite:

```bibtex
@article{zhang2025radir,
  title={RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining},
  author={Zhang, Tengfei and Zhao, Ziheng and Wu, Chaoyi and Zhou, Xiao and Zhang, Ya and Wang, Yangfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2503.04653},
  year={2025}
}
```
