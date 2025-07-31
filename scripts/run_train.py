import argparse
import datetime
import os
import yaml

import torch

from transformer_maskgit import CTViT, CTViT_Text_Fusion, CTViT_Temporal_Fusion, CTViT_Temporal_Fusion_butCLS
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, CTCLIP_NO_IMG_LATENT, CTCLIP_Tengfei_CrossAttn, CTCLIP_Earyly_Fusion, CTCLIP_Temporal_Fusion, CTCLIP_Temporal_Fusion_butCLS
from CTCLIPTrainer import CTClipTrainer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--config_file', type=str, help='Path to the config file (YAML format)')
    parser.add_argument('--fusion_module', type=str, help='crossattn, no_img_latent, tengfei_crossattn, early_fusion, temporal_fusion, temporal_fusion_butCLS')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--allow_partial_load', action='store_true')
    parser.add_argument('--evaluate_before_train', action='store_true')
    parser.add_argument('--use_triplet_loss', type=float, default=1)
    parser.add_argument('--use_infoNCE_loss', type=float, default=0)
    parser.add_argument('--freeze_pretrained_encoders', action='store_true')
    parser.add_argument('--open_partial_pretrained_encoders', action='store_true')
    parser.add_argument('--open_pretrained_encoders', action='store_true')
    parser.add_argument('--open_vision_encoder', action='store_true')
    parser.add_argument('--triplet_loss_margin',type=float, default=0.1)
    parser.add_argument('--positive_distance_threshold',type=float, default=0.25, help='triplet中ij>ik超过此阈值时拉大距离')
    parser.add_argument('--positive_threshold',type=float, default=0.75, help='infoNCE中大于此阈值的都视为false negative & triplet中大于此阈值才可以被作为positive')
    parser.add_argument('--negative_threshold',type=float, default=1, help='triplet中小于此阈值才可以被作为negative')
    parser.add_argument('--anatomy_filter', type=str, nargs='+', default=['pancreas', 'gallbladder', 'spleen', 'kidney'])
    parser.add_argument('--data_train_jsonl', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/train.jsonl')
    parser.add_argument('--data_valid_jsonl', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/test.jsonl')
    parser.add_argument('--data_train_csv_dir', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/train_entity')
    parser.add_argument('--data_valid_csv_dir', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/val_entity')
    parser.add_argument('--data_train_npy_dir', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/train_ratescore')
    parser.add_argument('--data_valid_npy_dir', type=str, default='/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/data/tengfei_proj/val_ratescore')
    parser.add_argument('--local_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, nargs='+', default=1e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--num_train_steps', type=int, nargs='+', default=100000)
    parser.add_argument('--warmup_steps', type=int, nargs='+', default=10000)
    parser.add_argument('--save_results_every', type=int, default=10000)
    parser.add_argument('--save_model_every', type=int, default=10000)
    parser.add_argument('--shuffle_train_samples_every', type=int, default=100000)
    parser.add_argument('--valid_max_samples', type=int, default=2000)  # 2k, allow all valid samples
    parser.add_argument('--train_max_samples', type=int, default=12000) # 24k -> 12k
    config = parser.parse_args()
    
    if config.config_file:
        with open(config.config_file, 'r') as f:
            loaded_config_dict = yaml.safe_load(f)
        for key, value in loaded_config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if os.path.exists('/DB/data/haoningwu-1/zihengzhao'):
        results_folder = f'/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/{config.name}'
        tokenizer = BertTokenizer.from_pretrained('/DB/data/haoningwu-1/zihengzhao/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        text_encoder = BertModel.from_pretrained("/DB/data/haoningwu-1/zihengzhao/BiomedVLP-CXR-BERT-specialized")
    else:
        results_folder = f'/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/CT-Conditional-Image-Retrieval/log/{config.name}'
        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    if config.fusion_module == 'early_fusion':
        image_encoder = CTViT_Text_Fusion(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
    elif config.fusion_module == 'temporal_fusion':
        image_encoder = CTViT_Temporal_Fusion(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
    elif config.fusion_module == 'temporal_fusion_butCLS':
        image_encoder = CTViT_Temporal_Fusion_butCLS(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
    else:
        image_encoder = CTViT(
            dim = 512,
            codebook_size = 8192,
            image_size = 480,
            patch_size = 20,
            temporal_patch_size = 10,
            spatial_depth = 4,
            temporal_depth = 4,
            dim_head = 32,
            heads = 8
        )
    #dim_image = 131072,

    if config.fusion_module == 'no_img_latent':
        clip = CTCLIP_NO_IMG_LATENT(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False
        )
    elif config.fusion_module == 'crossattn':
        clip = CTCLIP(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False
        )
    elif config.fusion_module == 'tengfei_crossattn':
        clip = CTCLIP_Tengfei_CrossAttn(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False
        )
    elif config.fusion_module == 'early_fusion':
        clip = CTCLIP_Earyly_Fusion(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False
        )
    elif config.fusion_module == 'temporal_fusion':
        clip = CTCLIP_Temporal_Fusion(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            triplet_loss_margin = config.triplet_loss_margin,
        )
    elif config.fusion_module == 'temporal_fusion_butCLS':
        clip = CTCLIP_Temporal_Fusion_butCLS(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = config.use_triplet_loss,
            use_infoNCE_loss = config.use_infoNCE_loss,
            positive_threshold = config.positive_threshold,
            negative_threshold = config.negative_threshold,
            positive_distance_threshold = config.positive_distance_threshold,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            triplet_loss_margin = config.triplet_loss_margin,
        )
    
    assert sum([
        config.open_pretrained_encoders,
        config.open_partial_pretrained_encoders,
        config.freeze_pretrained_encoders,
        config.open_vision_encoder
    ]) == 1, "Exactly one of open_pretrained_encoders, open_partial_pretrained_encoders, freeze_pretrained_encoders must be True"
    
    if config.freeze_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Freeze encoders')
        clip.freeze_text_encoder()
        clip.freeze_vision_encoder()
    elif config.open_partial_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Partially open encoders')
        clip.open_partial_text_encoder()
        clip.open_partial_vision_encoder()
    elif config.open_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Open encoders')
        clip.open_text_encoder()
        clip.open_vision_encoder()
    elif config.open_vision_encoder:
        clip.freeze_text_encoder()
        clip.open_vision_encoder()
    
    trainer = CTClipTrainer(
        clip,
        tokenizer = tokenizer,
        data_train_jsonl = config.data_train_jsonl,
        data_valid_jsonl = config.data_valid_jsonl,
        data_train_npy_dir = config.data_train_npy_dir,
        data_valid_npy_dir = config.data_valid_npy_dir,
        data_train_csv_dir = config.data_train_csv_dir,
        data_valid_csv_dir = config.data_valid_csv_dir,
        anatomy_filter = config.anatomy_filter,
        positive_threshold = config.positive_threshold,
        negative_threshold = config.negative_threshold,
        batch_size = config.batch_size,
        local_batch_size = config.local_batch_size,
        results_folder = results_folder,
        lr = config.lr,
        num_train_steps = config.num_train_steps,
        warmup_steps = config.warmup_steps,
        current_step = 1,
        save_results_every = config.save_results_every,
        save_model_every = config.save_model_every,
        shuffle_train_samples_every = config.shuffle_train_samples_every,
        num_workers = config.num_workers,
        pin_memory = config.pin_memory,
        train_max_samples=config.train_max_samples,
        valid_max_samples=config.valid_max_samples
    )
    
    # record configs for reproducibility
    f_path = os.path.join(results_folder, 'log.txt')
    if int(os.environ.get("RANK", 0)) == 0:
        with open(f_path, 'a') as f:
            # set exp time
            SHA_TZ = datetime.timezone(datetime.timedelta(hours=8),
                                name='Asia/Shanghai')   
            utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
            exp_time = f'{beijing_now.year}-{beijing_now.month}-{beijing_now.day}-{beijing_now.hour}-{beijing_now.minute}'
            f.write(f'{exp_time} \n Configs :\n')
            configDict = config.__dict__
            for eachArg, value in configDict.items():
                f.write(eachArg + ' : ' + str(value) + '\n')
            f.write('\n')
        print(f'Log at {results_folder}')

    if config.checkpoint is not None:
        if config.resume_training:
            trainer.resume_training(config.checkpoint)
        else:
            trainer.load_checkpoint(config.checkpoint, config.allow_partial_load)
            
    if int(os.environ.get("RANK", 0)) == 0:
        print('** MODEL ** The following parameters are FROZEN:')
        for name, param in clip.named_parameters():
            if not param.requires_grad:
                print(name)
        print('** MODEL ** The following parameters are TRAINABLE:')
        for name, param in clip.named_parameters():
            if param.requires_grad:
                print(name)
            
        print(f'Start Training')
    
    trainer.train(config.evaluate_before_train)
