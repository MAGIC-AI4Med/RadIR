# RadIR

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv)](https://www.arxiv.org/abs/2503.04653)  
[![HF](https://img.shields.io/badge/🤗-Data-yellow)](https://huggingface.co/datasets/zzh99/RadIR)

This is the official repository for the paper:  
**"RadIR: A Scalable Framework for Multi-Grained Medical Image Retrieval via Radiology Report Mining"**  
[(arXiv link)](https://www.arxiv.org/abs/2503.04653)

The MIMIC-IR and CTRATE-IR datasets are available at our [Hugging Face repository](https://huggingface.co/datasets/zzh99/RadIR).  
Model weights can be found at: [https://huggingface.co/timeseed/RadIR](https://huggingface.co/timeseed/RadIR)

> ⏳ Training and evaluation code will be released soon.

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

Download the model weights and verify the paths before running.

The following example shows how to extract:
- Unconditional image and text embeddings
- Anatomy-conditioned fused image features

```python
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from radir import RADIR
from data_process import load_2d_image_to_tensor, load_3d_image_to_tensor
import torch

MODALITY_DICT = {'CT': 0, 'CXR': 1}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized").to(device)
    
    # Initialize image encoder
    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=8, temporal_depth=6,
        cls_depth=4, dim_head=32, heads=8
    ).to(device)
    
    # Initialize RadIR model
    rad_ir = RADIR(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dim_text=768,
        dim_image=512,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    ).to(device)

    # Load checkpoint
    checkpoint_path = 'checkpoints/RadIR.pt'
    rad_ir.load(checkpoint_path)
    rad_ir.eval()
    
    # Example data
    cxr_images = ['example/1.jpg', 'example/2.jpg']
    cxr_reports = [
        "Lateral view somewhat limited due to overlying motion artifact...",
        "Left PICC tip is seen terminating in the region..."
    ]
    anatomy_condition = ['right lung']
    
    with torch.no_grad():
        # Process images
        image_tensors = [load_2d_image_to_tensor(path) for path in cxr_images]
        batched_images = torch.stack(image_tensors, dim=0).to(device)  # [2, 1, 1, 480, 480]
        
        # Tokenize reports
        text_tokens = tokenizer(
            cxr_reports,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        modal_indices = torch.tensor([MODALITY_DICT['CXR']] * len(cxr_images)).to(device)
        
        # Extract unconditional embeddings
        image_embeddings, text_embeddings, _, _ = rad_ir(
            text_tokens, image=batched_images, device=device,
            is_condition=False, return_latents=True,
            modal_indexs=modal_indices, modal_embedding=True
        )
        
        print(f"Image embedding shape: {image_embeddings.shape}")  # [B, 512]
        print(f"Text embedding shape: {text_embeddings.shape}")    # [B, 512]
        
        # Extract condition-specific features (e.g., right lung)
        con_batch_image = batched_images.unsqueeze(0)  # [1, 2, 1, 1, 480, 480]
        con_modal_indices = torch.tensor([MODALITY_DICT['CXR']] * len(con_batch_image)).to(device)
        
        anatomy_condition_tokens = tokenizer(
            anatomy_condition,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        _, _, condition_feature, _ = rad_ir(
            anatomy_condition_tokens,
            image=con_batch_image,
            device=device,
            is_condition=True,
            return_latents=True,
            modal_indexs=con_modal_indices,
            modal_embedding=True
        )
        
        print(f"Condition feature shape: {condition_feature.shape}")  # [1, 2, 512]
```

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

---
**Note**: This is an early release. Full training and evaluation scripts will be published shortly.