from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from radir import RADIR
from data_process import load_2d_image_to_tensor, load_3d_image_to_tensor
import torch

MODALITY_DICT = {'CT': 0, 'CXR': 1}

if __name__ == '__main__':
    # Determine best available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized").to(device)
    
    # Initialize image encoder
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=8,
        temporal_depth=6,
        cls_depth=4,
        dim_head=32,
        heads=8
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
    rad_ir.eval()  # Set to evaluation mode
    
    # Test examples
    cxr_images = ['example/1.jpg', 'example/2.jpg']
    cxr_reports = [
        "Lateral view somewhat limited due to overlying motion artifact. The lungs are low in volume. There is no focal airspace consolidation to suggest pneumonia. A 1.2-cm calcified granuloma just below the medial aspect of the right hemidiaphragm is unchanged from prior study. No pleural effusions or pulmonary edema. There is no pneumothorax. The inferior sternotomy wire is fractured but unchanged. Surgical clips and vascular markers in the thorax are related to prior CABG surgery. No evidence of acute cardiopulmonary process.",
        "Left PICC tip is seen terminating in the region of the distal left brachiocephalic vein. Tracheostomy tube is in unchanged standard position. The heart is moderately enlarged. Marked calcification of the aortic knob is again present. Mild pulmonary vascular congestion is similar. Bibasilar streaky airspace opacities are minimally improved. Previously noted left pleural effusion appears to have resolved. No pneumothorax is identified. Percutaneous gastrostomy tube is seen in the left upper quadrant. 1. Left PICC tip appears to terminate in the distal left brachiocephalic vein. 2. Mild pulmonary vascular congestion. 3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis. Interval resolution of the left pleural effusion."
    ]
    anatomy_condition = ['right lung']
    
    # Extract unconditional image-text embeddings
    with torch.no_grad():
        # Process images
        image_tensors = [load_2d_image_to_tensor(image_path) for image_path in cxr_images]
        batched_images = torch.stack(image_tensors, dim=0).to(device)  # Shape: [B, C, D, H, W] -> [2, 1, 1, 480, 480]
        
        # Process text
        text_tokens = tokenizer(
            cxr_reports,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        print(text_tokens)
        
        # Set modality indicator for CXR
        modal_indices = torch.tensor([MODALITY_DICT['CXR']] * len(cxr_images)).to(device)
        
        # Forward pass to get embeddings
        image_embeddings, text_embeddings, _, _ = rad_ir(
            text_tokens, 
            image=batched_images,
            device=device,
            is_condition=False,  # Indicate unconditional scenario
            return_latents=True,
            modal_indexs=modal_indices,
            modal_embedding=True
        )
        
        print(f"Image embedding shape: {image_embeddings.shape}")  # [B, 512]
        print(f"Text embedding shape: {text_embeddings.shape}")    # [B, 512]
        
        # Extract condition-specific features (focused on right lung)
        con_batch_image = batched_images.unsqueeze(0)  # Shape: [B, local_B, C, D, H, W] -> [1, 2, 1, 1, 480, 480]
        
        # Set modality indicator for conditions
        con_modal_indices = torch.tensor([MODALITY_DICT['CXR']] * len(con_batch_image)).to(device)
        
        # Tokenize anatomical condition
        anatomy_condition_tokens = tokenizer(
            anatomy_condition,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # Extract conditional features
        _, _, condition_feature, _ = rad_ir(
            anatomy_condition_tokens,
            image=con_batch_image,
            device=device,
            is_condition=True,  # Indicate conditional scenario
            return_latents=True,
            modal_indexs=con_modal_indices,
            modal_embedding=True
        )
        
        print(f"Condition feature shape: {condition_feature.shape}")  # [1, 2, 512]