from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from radir import RADIR
from data_process import load_2d_image_to_tensor, load_3d_image_to_tensor
import torch

modality_dict = {'CT':0,'CXR':1}
if __name__ == '__main__':
    # # 确定最佳可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载tokenizer和文本编码器
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder = text_encoder.to(device)  # 将文本编码器移至设备
    
    # 初始化图像编码器
    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 8,
        temporal_depth = 6,
        cls_depth = 4,
        dim_head = 32,
        heads = 8
    ).to(device)  # 将图像编码器移至设备
    
    # 初始化RadIR模型
    Rad_IR = RADIR(
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        dim_text = 768,
        dim_image = 512,
        dim_latent = 512,
        extra_latent_projection = False,
        use_mlm = False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    ).to(device)  # 将整个模型移至设备

    # 加载checkpoint
    checkpoint_path = '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/log/CXR_CT_Uni_CXRCT_only_fusion_all_93_use_info——10——test/CTClip.8000.pt'
    Rad_IR.load(checkpoint_path)
    
    # 设置为评估模式
    Rad_IR.eval()
    
    # 测试示例
    CXR_image_example = ['example/1.jpg', 'example/2.jpg']
    CXR_text_example = [
        "Lateral view somewhat limited due to overlying motion artifact. The lungs are low in volume. There is no focal airspace consolidation to suggest pneumonia. A 1.2-cm calcified granuloma just below the medial aspect of the right hemidiaphragm is unchanged from prior study. No pleural effusions or pulmonary edema. There is no pneumothorax. The inferior sternotomy wire is fractured but unchanged. Surgical clips and vascular markers in the thorax are related to prior CABG surgery. No evidence of acute cardiopulmonary process.",
        "Left PICC tip is seen terminating in the region of the distal left brachiocephalic vein. Tracheostomy tube is in unchanged standard position. The heart is moderately enlarged. Marked calcification of the aortic knob is again present. Mild pulmonary vascular congestion is similar. Bibasilar streaky airspace opacities are minimally improved. Previously noted left pleural effusion appears to have resolved. No pneumothorax is identified. Percutaneous gastrostomy tube is seen in the left upper quadrant. 1. Left PICC tip appears to terminate in the distal left brachiocephalic vein. 2. Mild pulmonary vascular congestion. 3. Interval improvement in aeration of the lung bases with residual streaky opacity likely reflective of atelectasis. Interval resolution of the left pleural effusion."
    ]
    anatomy_condition = ['right lung']
    
    # 无条件的图文embedding提取
    with torch.no_grad():  # 在评估模式下不需要计算梯度
        # 处理图像
        image_tensors = [load_2d_image_to_tensor(image_path) for image_path in CXR_image_example]
        batched_images = torch.stack(image_tensors, dim=0)  # 形状: [B, C, D, H, W] -> [2, 1, 1, 480, 480]
        batched_images = batched_images.to(device)  # 将图像移至设备
        
        # 处理文本
        text_tokens = tokenizer(
            CXR_text_example,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        print(text_tokens)
        
        modal_indexs = torch.tensor([modality_dict['CXR']] * len(CXR_image_example)).to(device)  # 提示模型当前模态为CXR
        # 前向传播获取embeddings
        image_embeddings, text_embeddings, _, _ = Rad_IR(
            text_tokens, 
            image=batched_images,
            device=device,      # 使用已定义的设备
            is_condition=False, # 指示当前为无条件的情况
            return_latents=True,
            modal_indexs=modal_indexs,
            modal_embedding=True
        )
        
        print(f"图像嵌入形状: {image_embeddings.shape}") # [B, 512]
        print(f"文本嵌入形状: {text_embeddings.shape}")  # [B, 512]
        

        
        # 聚焦于特定的解剖结构的有条件特征. right lung
        # 这里可以添加条件特征提取的代码
        con_batch_image = batched_images.unsqueeze(0)  # 形状: [B, local_B, C, D, H, W] -> [1, 2, 1, 1, 480, 480]
        
        # local_B表示同一个条件下的图像数量
        con_modal_indexs = torch.tensor([modality_dict['CXR']] * len(con_batch_image)).to(device) # 大小是 B
        anatomy_condition_tokens = tokenizer(           # 大小是B
            anatomy_condition,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        _,_, condition_feature, _ = Rad_IR(
            anatomy_condition_tokens,
            image=con_batch_image,
            device=device,
            is_condition=True, # 指示当前为有条件的情况
            return_latents=True,
            modal_indexs=con_modal_indexs,
            modal_embedding=True
        )
        print(f"条件特征形状: {condition_feature.shape}") # [1, 2, 512]