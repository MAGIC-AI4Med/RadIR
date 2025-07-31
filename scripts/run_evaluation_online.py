import argparse
from tqdm import tqdm
import pandas as pd
import os
import json
from pathlib import Path
from datetime import timedelta
import heapq

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
import numpy as np

from torch.utils.data import DataLoader

from ct_clip import CTCLIP
from retrieval_metric import hard_retrieval, soft_retrieval, compute_ndcg, hard_retrieval_exclude_self, hard_retrieval_exclude_self_return_details
from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval
from transformer_maskgit import CTViT, CTViT_Text_Fusion, CTViT_Temporal_Fusion, CTViT_Temporal_Fusion_butCLS
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, CTCLIP_NO_IMG_LATENT, CTCLIP_Tengfei_CrossAttn, CTCLIP_Earyly_Fusion, CTCLIP_Temporal_Fusion, CTCLIP_Temporal_Fusion_butCLS
from CTCLIPTrainer import CTClipTrainer

#
# This is code is to evaluate RadIR-ChestCT with multi-grained condition
#

def distribute_equal_sample_num(anatomy_samples, num_processes):
    """
    改进后的分配策略：
    1. 当anatomy数量 ≤ 进程数时：
    - 每个进程分配1个anatomy（按样本量降序）
    - 多余进程循环复用高样本量anatomy
    2. 当anatomy数量 > 进程数时：
    - 使用贪心算法均衡分配，使各进程总样本量接近
    """
    # 按样本量降序、名称升序排序
    sorted_anatomy = sorted(
        anatomy_samples.items(),
        key=lambda x: (-x[1], x[0])
    )
    num_anatomy = len(sorted_anatomy)
    
    # 情况1：anatomy数量 ≤ 进程数
    if num_anatomy <= num_processes:
        distribution = []
        for i in range(num_processes):
            # 循环使用高样本量anatomy
            anatomy_idx = i % num_anatomy
            anatomy_name = sorted_anatomy[anatomy_idx][0]
            distribution.append([anatomy_name])
    
        # 针对validation分配结果（多余进程置空）
        backup_dist = []
        for i in range(num_processes):
            if i < num_anatomy:
                anatomy_name = sorted_anatomy[i][0]
                backup_dist.append([anatomy_name])
            else:
                backup_dist.append([])
                
        return distribution, backup_dist
    
    # 情况2：anatomy数量 > 进程数
    else:
        # 初始化最小堆（总样本量，进程ID，anatomy列表）
        heap = []
        for i in range(num_processes):
            heapq.heappush(heap, (0, i, []))
        
        # 贪心分配每个anatomy
        for anatomy_name, samples in sorted_anatomy:
            total, proc_id, anatomy_list = heapq.heappop(heap)
            anatomy_list.append(anatomy_name)
            heapq.heappush(heap, (total + samples, proc_id, anatomy_list))
        
        # 提取分配结果并按进程ID排序
        distribution = [[] for _ in range(num_processes)]
        while heap:
            total, proc_id, anatomy_list = heapq.heappop(heap)
            distribution[proc_id] = anatomy_list
        
        return distribution, distribution

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str)    # make sure the index is consistent with the data_jsonl
    parser.add_argument('--anatomy_filter', type=str, nargs='+', default=['aorta', 'heart'])
    parser.add_argument('--topk', type=int, nargs='+', default=[3, 5, 10, 20, 50, 100])
    parser.add_argument('--valid_max_samples', type=int, default=10000)
    parser.add_argument('--fusion_module', type=str, default='temporal_fusion_butCLS')
    parser.add_argument('--checkpoint', type=str, default='/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/final15_Anatomy/final15_Anatomy_0.9_0.2_0.8_aug_sqrt(equal_anatomy_sample)/CTClip.400.pt')
    parser.add_argument('--allow_partial_load', action='store_true')
    parser.add_argument('--return_detail', action='store_true')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_valid_jsonl', type=str, default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/test.jsonl')
    parser.add_argument('--data_valid_csv_dir', type=str, default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_entity')
    parser.add_argument('--data_valid_npy_dir', type=str, default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_ratescore')
    config = parser.parse_args()
    
    # Prepare Model

    if os.path.exists('/DB/data/haoningwu-1/zihengzhao'):
        results_folder = f'/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/Evaluation_Results/{config.results_folder}'
        tokenizer = BertTokenizer.from_pretrained('/DB/data/haoningwu-1/zihengzhao/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        text_encoder = BertModel.from_pretrained("/DB/data/haoningwu-1/zihengzhao/BiomedVLP-CXR-BERT-specialized")
    else:
        results_folder = f'/mnt/petrelfs/zhaoziheng/CT-RATE-Related-Project/CT-RATE/CT-Conditional-Image-Retrieval/log/Evaluation_Results/{config.results_folder}'
        tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    Path(results_folder).mkdir(parents=True, exist_ok=True)

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

    if config.fusion_module == 'no_img_latent':
        clip = CTCLIP_NO_IMG_LATENT(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
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
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
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
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
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
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
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
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            triplet_loss_margin = 0.1,
        )
    elif config.fusion_module == 'temporal_fusion_butCLS':
        clip = CTCLIP_Temporal_Fusion_butCLS(  # Cross-Attn as feature fusion module will be initialized in CTCLIP 
            image_encoder = image_encoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            use_triplet_loss = 1,
            use_infoNCE_loss = 0,
            positive_threshold = 0.9,
            negative_threshold = 0.8,
            positive_distance_threshold = 0.2,
            dim_text = 768,
            dim_image = 294912,
            dim_latent = 512,
            extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
            use_mlm=False,
            downsample_image_embeds = False,
            use_all_token_embeds = False,
            triplet_loss_margin = 0.1,
        )
        
    path = Path(config.checkpoint)
    assert path.exists()
    pkg = torch.load(path, map_location='cpu')
    
    if config.allow_partial_load:
        model_dict = clip.state_dict()
        pkg_state_dict = pkg['model']  # 假设 pkg['model'] 是模型的状态字典

        # 检查差异
        unexpected_state_dict = [k for k in pkg_state_dict.keys() if k not in model_dict.keys()]
        missing_state_dict = [k for k in model_dict.keys() if k not in pkg_state_dict.keys()]
        unmatchd_state_dict = [k for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]

        # 加载部分参数
        state_dict = {k: v for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        clip.load_state_dict(model_dict)

        print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
        print(unexpected_state_dict)
        print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
        print(missing_state_dict)
        print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
        print(unmatchd_state_dict)
        print('** MODEL ** The following parameters are LOADED in:\n')
        print(state_dict.keys())
    else:
        clip.load_state_dict(pkg['model'])
        
    # Set DDP
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs])
    device = accelerator.device
    clip.to(device)
    
    # Prepare Data
    
    with open(f"/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/(all)anatomy_positive_elements_count.json", 'r') as f:
        anatomy_distribution = json.load(f)
    anatomy_samples = {anatomy:anatomy_distribution[anatomy]["train >0.9 links count"] for anatomy in config.anatomy_filter}   # pos 样本越多，采样越多
    
    # Replace the direct call to torch.distributed.get_rank() with a safe check.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()  # 获取当前进程的 rank
        world_size = torch.distributed.get_world_size()  # 获取总进程数
    else:
        rank = 0  # Default rank if not using distributed training
        world_size = 1
    _, distribution_valid = distribute_equal_sample_num(anatomy_samples, world_size)
    local_anatomy_filter_for_valid = distribution_valid[rank]
        
    valid_ds = Conditional_CTReportDataset_Eval(
            jsonl_file=config.data_valid_jsonl, 
            csv_file_dir=config.data_valid_csv_dir, 
            npy_file_dir=config.data_valid_npy_dir, 
            anatomy_filter=local_anatomy_filter_for_valid,
            max_samples=config.valid_max_samples
    )
    
    anatomy2id_ls =valid_ds.get_anatomy2id_ls()

    valid_dl = DataLoader(
            valid_ds,
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            shuffle = False,
            pin_memory = config.pin_memory,
    )
    
    # Begin Eval in DDP Mode
    
    with torch.no_grad():
            
        clip.eval()
            
        all_metrics = {
            'NDCG@3': {},
            'NDCG@5': {},
            'NDCG@10': {},
            'NDCG@20': {},
            'NDCG@50': {},
            'NDCG@100': {},
            'Recall@3': {},
            'Recall@5': {},
            'Recall@10': {},
            'Recall@20': {},
            'Recall@50': {},
            'Recall@100': {},
            'RateScore@3': {},
            'RateScore@5': {},
            'RateScore@10': {},
            'RateScore@20': {},
            'RateScore@50': {},
            'RateScore@100': {},
            'UpperBound_RateScore@3': {},
            'UpperBound_RateScore@5': {},
            'UpperBound_RateScore@10': {},
            'UpperBound_RateScore@20': {},
            'UpperBound_RateScore@50': {},
            'UpperBound_RateScore@100': {},
        }
        
        fused_latents_all = {anatomy:torch.zeros(len(anatomy2id_ls[anatomy]), 512) for anatomy in local_anatomy_filter_for_valid}   
        # 每个anatomy的fused_latents顺序与dataset中的sample id顺序严格一致
        # 每个anatomy的fused_latents的shape是 (N, 512)，N是这个anatomy的样本数
        
        # First derive the fusion latent for each anatomy and each sample
        for video, anatomy_ls, sample_id_ls in tqdm(valid_dl, desc=f"Rank {rank}"):
            # video: B 1 d h w
            video = video.to(device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
            anatomy_ls = list(anatomy_ls)
            sample_id_ls = list(sample_id_ls)
            text_tokens=tokenizer(anatomy_ls, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
            _, _, fused_latents, tmp = clip(text_tokens, video, return_latents=True, device=device) # B 1 Dim
            fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
            for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
                sample_index = anatomy2id_ls[anatomy].index(sample_id)
                fused_latents_all[anatomy][sample_index] = fused_latents[i]
                    
        # Now lets calculate the similarity matrix for each anatomy
        for anatomy, fused_latents in fused_latents_all.items():
            
            fused_latents = fused_latents.to(device)
            image_to_image = torch.einsum('m d, n d -> m n', fused_latents, fused_latents)
            similarity_tab = valid_ds.get_similarity_table(anatomy)
            similarity_tab = torch.tensor(similarity_tab)
            similarity_tab = similarity_tab.to(dtype=torch.float32) * 0.01    # 0~100 uint8 -> 0~1 float32

            assert image_to_image.shape == similarity_tab.shape, f'image_to_image {image_to_image.shape} != similarity_tab {similarity_tab.shape} != {len(anatomy2id_ls[anatomy])}'
            
            # now calculate the ndcg
            index_list = [i for i in range(similarity_tab.shape[0])]
            ndcg_scores = compute_ndcg(image_to_image, similarity_tab, index_list, k=config.topk)
            for k, v in zip(config.topk, ndcg_scores):
                all_metrics[f'NDCG@{k}'][anatomy] = v*100
            
            # average similarity
            pred_score, upperbound_score = soft_retrieval(image_to_image, similarity_tab, k=config.topk)   # {'avg_similarity@{k}': xxx, ...}
            for k, v in zip(config.topk, pred_score):
                all_metrics[f'RateScore@{k}'][anatomy] = v*100
            for k, v in zip(config.topk, upperbound_score):
                all_metrics[f'UpperBound_RateScore@{k}'][anatomy] = v*100
            
            # hard image-image retrieval
            if config.return_detail:
                # save recall results of each sample in a json
                details = hard_retrieval_exclude_self_return_details(image_to_image, similarity_tab, index_list, k=config.topk, return_detail=True)
                samplewise_recall = details['row_recall']    # n
                samplewise_topk_results = details['prediction_topk_indices']    # n k
                samplewise_topk_scores = details['prediction_topk_scores']      # n k
                samplewise_topk_gt_results = details['gt_topk_indices']         # n k
                samplewise_topk_gt_scores = details['gt_topk_scores']           # n k
                id_ls = anatomy2id_ls[anatomy]  # n
                detailed_results = {}
                for query_idx, query_id in enumerate(id_ls):
                    topk_results_unaligned = samplewise_topk_results[query_idx].tolist()
                    topk_results = []
                    for idx in topk_results_unaligned:
                        if idx >= query_idx:
                            topk_results.append(id_ls[idx+1])
                        else:
                            topk_results.append(id_ls[idx])
                    # topk_results = [id_ls[idx] for row_idx, idx in enumerate(topk_results)]
                    topk_scores = samplewise_topk_scores[query_idx].tolist()
                    gt_topk_results_unaligned = samplewise_topk_gt_results[query_idx].tolist()
                    gt_topk_results = []
                    for idx in gt_topk_results_unaligned:
                        if idx >= query_idx:
                            gt_topk_results.append(id_ls[idx+1])
                        else:
                            gt_topk_results.append(id_ls[idx])
                    gt_topk_scores = samplewise_topk_gt_scores[query_idx].tolist()
                    detailed_results[query_id] = {
                        f'recall@{config.topk}': samplewise_recall[query_idx],
                        'prediction_topk': [list(item) for item in zip(topk_results, topk_scores)],
                        'gt_topk': [list(item) for item in zip(gt_topk_results, gt_topk_scores)],
                    }
                with open(f'{results_folder}/{anatomy}.json', 'w') as f:
                    json.dump(detailed_results, f, indent=4,
                              default=lambda o: float(o) if isinstance(o, np.float32) else o)
                recall_scores = details['recall']
            else:
                recall_scores = hard_retrieval_exclude_self(image_to_image, similarity_tab, index_list, k=config.topk)
            if recall_scores == -1:
                print(f'{anatomy} has no positive samples!')
                for k in config.topk:
                    all_metrics[f'Recall@{k}'][anatomy] = -1
            else:
                for k, v in zip(config.topk, recall_scores):
                    all_metrics[f'Recall@{k}'][anatomy] = v*100
                    
        # 清理显存，释放不再需要的变量
        del image_to_image, fused_latents_all, similarity_tab
        torch.cuda.empty_cache()  # 释放显存缓存

        if torch.distributed.is_initialized():
            # Gather all_metrics dicts from all processes into a list on each process
            world_size = torch.distributed.get_world_size()
            gathered_metrics = [None for _ in range(world_size)]
            
            # +++ 新增：构建包含损失统计量的数据结构 +++
            gathered_data = {
                'metrics': all_metrics,
            }
            
            # 执行收集（替换原 all_gather_object 调用）
            torch.distributed.all_gather_object(gathered_metrics, gathered_data)  # +++ 修改行 +++

            # On the main process, merge the dictionaries
            if accelerator.is_local_main_process:
                
                print(f'** EVAL ** Gather Done!')
                
                # +++ 修改信息输出部分 +++
                info = ''

                merged_metrics = {}
                for metric_name in all_metrics.keys():
                    merged_metrics[metric_name] = {}
                for gathered_data in gathered_metrics:
                    for metric_name, subdict in gathered_data['metrics'].items():   # metric_name: 'Recall@5' ...
                        for key, value in subdict.items():  # pancreas: 0.8
                            merged_metrics[metric_name][key] = value
                all_metrics = merged_metrics
                
                for metrics_name in ['NDCG', 'Recall', 'RateScore', 'UpperBound_RateScore']:
                    for k in config.topk:
                        avg_results = sum(all_metrics[f'{metrics_name}@{k}'].values()) / len(all_metrics[f'{metrics_name}@{k}'])
                        info += f' {metrics_name}@{k} {avg_results} |'
                        
                        write_info = info   # the following details will be written but not displayed
                        
                        for key, value in all_metrics[f'{metrics_name}@{k}'].items():
                            write_info += f"{key}_{metrics_name}@{k}: {value} | "
                        
                info += '\n'
                write_info += '\n'
                print(info)
                with open(f'{results_folder}/log.txt', 'a') as f:
                    f.write(info)
                    
                with open(f'{results_folder}/results.json', 'w') as f:
                    json.dump(all_metrics, f, indent=4)
                    
                print(f'** EVAL ** Log Done!')
    

    
    
