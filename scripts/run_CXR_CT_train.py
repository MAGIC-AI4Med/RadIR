import argparse
import datetime
import os
import yaml

import torch

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from radir import RADIR

from RadIRTrainer_CXR_CT import RadIRTrainer

import json
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',required=True, type=str)
    parser.add_argument('--config_file', type=str, help='Path to the config file (YAML format)')

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--allow_partial_load', action='store_true')
    parser.add_argument('--evaluate_before_train', action='store_true')
    parser.add_argument('--use_triplet_loss', type=float, default=1)
    parser.add_argument('--use_infoNCE_loss', type=float, default=0)
    parser.add_argument('--use_image2image_loss', type=float, default=1)
    parser.add_argument('--use_uncon_triplet_loss', type=float, default=1)
    parser.add_argument('--use_uncon_infoNCE_loss', type=float, default=1)
    parser.add_argument('--freeze_pretrained_encoders', action='store_true')
    parser.add_argument('--open_partial_pretrained_encoders', action='store_true')
    parser.add_argument('--open_pretrained_encoders', action='store_true')
    parser.add_argument('--open_vision_encoder', action='store_true')
    parser.add_argument('--open_fusion_module', action='store_true')
    parser.add_argument('--freeze_fusion_module', action='store_true')
    parser.add_argument('--triplet_loss_margin',type=float, default=0.1)
    parser.add_argument('--positive_distance_threshold',type=float, default=0.25, help='triplet中ij>ik超过此阈值时拉大距离')
    parser.add_argument('--positive_threshold',type=float, default=0.75, help='infoNCE中大于此阈值的都视为false negative & triplet中大于此阈值才可以被作为positive')
    parser.add_argument('--negative_threshold',type=float, default=1, help='triplet中小于此阈值才可以被作为negative')
    parser.add_argument('--anatomy_filter', type=str, default='[["left hemidiaphragm","airspace"]["right rib","thyroid gland"]]')
    parser.add_argument('--dataset_names',type=str, nargs='+', default=['MIMIC-CXR','CT_RATE'])
    parser.add_argument('--modality',type=str, nargs='+',default=['2D', '3D'])
    parser.add_argument('--results_folder', type=str, default='/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/log')

    parser.add_argument('--data_train_jsonl', nargs='+',type=str, required=True)
    parser.add_argument('--data_valid_jsonl', nargs='+',type=str, required=True)
    parser.add_argument('--data_train_csv_dir', nargs='+',type=str, required=True)
    parser.add_argument('--data_valid_csv_dir', nargs='+',type=str, required=True)
    parser.add_argument('--data_train_npy_dir', nargs='+',type=str, required=True)
    parser.add_argument('--data_valid_npy_dir', nargs='+',type=str, required=True)
    parser.add_argument('--local_batch_size', nargs='+',type=int, default=[64,8])
    parser.add_argument('--batch_size', nargs='+',type=int, default=[4,1])
    parser.add_argument('--train_valid_length_',type=int, default=2400)
    # Additional unconstrained dataset parameters
    parser.add_argument('--uncon_batch_size', nargs='+', type=int, help='Batch size for unconstrained datasets')
    parser.add_argument('--uncon_batch_size_valid', nargs='+', type=int, help='Batch size for validation of unconstrained datasets')
    parser.add_argument('--uncon_soft_label', action='store_true', help='Use soft labels for unconstrained datasets')
    # 表示文件夹的路径，并不是实际文件的路径
    parser.add_argument('--uncon_similarity_lookup_table_train', nargs='+', type=str, help='Similarity lookup tables for unconstrained training datasets')
    parser.add_argument('--uncon_similarity_lookup_table_valid', nargs='+', type=str, help='Similarity lookup tables for unconstrained validation datasets')
    parser.add_argument('--uncon_data_train_jsonl', nargs='+', type=str, help='JSONL files for unconstrained training datasets')
    parser.add_argument('--uncon_data_valid_jsonl', nargs='+', type=str, help='JSONL files for unconstrained validation datasets')
    parser.add_argument('--uncon_dataset_names', nargs='+', type=str, help='Names of unconstrained datasets')
    parser.add_argument('--uncon_train_filter', type=str, help='2D list of training filters for unconstrained datasets')
    parser.add_argument('--uncon_valid', nargs='+', type=str, help='List of validation dataset names for unconstrained datasets')
    parser.add_argument('--uncon_modality', nargs='+', type=str, help='Modality types (2D/3D) for unconstrained datasets')
    parser.add_argument('--stage1', action='store_true', help='Whether to run stage 1 training')
    parser.add_argument('--stage2', action='store_true', help='Whether to run stage 2 training')
    parser.add_argument('--modal_embedding', action='store_true', help='Whether to use modal embdding')
    parser.add_argument('--train_no_aug', action='store_true', help='Whether to use modal embdding')
    parser.add_argument('--infonce_temp', action='store_true', help='Whether to use infonce temperature')
    parser.add_argument('--lr', type=float, nargs='+', default=1e-4)
    parser.add_argument('--uncon_num_workers', type=int, default=4)
    parser.add_argument('--con_num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--num_train_steps', type=int, nargs='+', default=100000)
    parser.add_argument('--warmup_steps', type=int, nargs='+', default=10000)
    parser.add_argument('--save_results_every', type=int, default=10000)
    parser.add_argument('--save_model_every', type=int, default=10000)
    parser.add_argument('--shuffle_train_samples_every', type=int, default=100000)
    parser.add_argument('--valid_max_samples', type=int, default=2000)  # 2k, allow all valid samples
    parser.add_argument('--train_max_samples', type=int, default=12000) # 24k -> 12k
    config = parser.parse_args()
    config.anatomy_filter = json.loads(config.anatomy_filter)  # convert string to list
    if config.config_file:
        with open(config.config_file, 'r') as f:
            loaded_config_dict = yaml.safe_load(f)
        for key, value in loaded_config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    results_folder = f'{config.results_folder}/{config.name}'
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

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
    )


    Rad_IR = RADIR(  
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        use_triplet_loss = config.use_triplet_loss,
        use_infoNCE_loss = config.use_infoNCE_loss,
        use_uncon_triplet_loss = config.use_uncon_triplet_loss,
        use_uncon_infoNCE_loss = config.use_uncon_infoNCE_loss,
        use_image2image_loss = config.use_image2image_loss,
        infonce_temp = config.infonce_temp,
        positive_threshold = config.positive_threshold,
        negative_threshold = config.negative_threshold,
        positive_distance_threshold = config.positive_distance_threshold,
        dim_text = 768,
        dim_image = 512,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    )
    
    assert sum([
        config.open_pretrained_encoders,
        config.open_partial_pretrained_encoders,
        config.freeze_pretrained_encoders,
        config.open_vision_encoder
    ]) == 1, "Exactly one of open_pretrained_encoders, open_partial_pretrained_encoders, freeze_pretrained_encoders must be True"
    
    
    assert sum([
        config.open_fusion_module,
        config.freeze_fusion_module
    ]) == 1, "Exactly one of open_fusion_module, freeze_fusion_module must be True"
    
    if config.freeze_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Freeze encoders')
        Rad_IR.freeze_text_encoder()
        Rad_IR.freeze_vision_encoder()
    elif config.open_partial_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Partially open encoders')
        Rad_IR.open_partial_text_encoder()
        Rad_IR.open_partial_vision_encoder()
    elif config.open_pretrained_encoders:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Open encoders')
        Rad_IR.open_text_encoder()
        Rad_IR.open_vision_encoder()
    elif config.open_vision_encoder:
        Rad_IR.freeze_text_encoder()
        Rad_IR.open_vision_encoder()
    
    if config.open_fusion_module:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Open fusion module')
        Rad_IR.open_fusion_module()
    elif config.freeze_fusion_module:
        if int(os.environ.get("RANK", 0)) == 0:
            print('** MODEL ** Freeze fusion module')
        Rad_IR.freeze_fusion_module()
        
    # 输出pid
    print(f"最开始 pid={os.getpid()} ",'rank', torch.distributed.get_rank() if torch.distributed.is_initialized() else 'not distributed')
    if config.stage1 and config.stage2:
        print('同时开始两阶段的训练')
    elif config.stage1:
        print('开始第一阶段的训练')
    elif config.stage2:
        print('开始第二阶段的训练')
    else:
        raise ValueError('At least one of stage1 or stage2 must be True')
    trainer = RadIRTrainer(
        Rad_IR,
        tokenizer = tokenizer,
        data_train_jsonl = config.data_train_jsonl,
        data_valid_jsonl = config.data_valid_jsonl,
        data_train_npy_dir = config.data_train_npy_dir,
        data_valid_npy_dir = config.data_valid_npy_dir,
        data_train_csv_dir = config.data_train_csv_dir,
        data_valid_csv_dir = config.data_valid_csv_dir,
        anatomy_filter = config.anatomy_filter,
        dataset_names = config.dataset_names,
        modality = config.modality,
        positive_threshold = config.positive_threshold,
        negative_threshold = config.negative_threshold,
        batch_size = config.batch_size,
        local_batch_size = config.local_batch_size,
        stage1= config.stage1,
        stage2= config.stage2,
        uncon_batch_size= config.uncon_batch_size,
        uncon_batch_size_valid= config.uncon_batch_size_valid,
        uncon_soft_label = config.uncon_soft_label,
        uncon_similarity_lookup_table_train = config.uncon_similarity_lookup_table_train,
        uncon_similarity_lookup_table_valid = config.uncon_similarity_lookup_table_valid,
        uncon_data_train_jsonl = config.uncon_data_train_jsonl,
        uncon_data_valid_jsonl = config.uncon_data_valid_jsonl,
        uncon_dataset_names = config.uncon_dataset_names,
        uncon_train_filter = json.loads(config.uncon_train_filter) if config.uncon_train_filter else None,
        uncon_valid = config.uncon_valid,
        uncon_modality = config.uncon_modality,
        modal_embedding = config.modal_embedding,
        train_no_aug = config.train_no_aug,
        train_valid_length_ = config.train_valid_length_,
        results_folder = results_folder,
        lr = config.lr,
        num_train_steps = config.num_train_steps,
        warmup_steps = config.warmup_steps,
        current_step = 1,
        save_results_every = config.save_results_every,
        save_model_every = config.save_model_every,
        shuffle_train_samples_every = config.shuffle_train_samples_every,
        con_num_workers = config.con_num_workers,
        uncon_num_workers= config.uncon_num_workers,
        pin_memory = config.pin_memory,
        train_max_samples=config.train_max_samples,
        valid_max_samples=config.valid_max_samples
    )
    
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
        for name, param in Rad_IR.named_parameters():
            if not param.requires_grad:
                print(name)
        print('** MODEL ** The following parameters are TRAINABLE:')
        for name, param in Rad_IR.named_parameters():
            if param.requires_grad:
                print(name)
            
        print(f'Start Training')
    
    trainer.train(config.evaluate_before_train)
