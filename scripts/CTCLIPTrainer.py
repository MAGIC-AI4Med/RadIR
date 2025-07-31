from pathlib import Path
import json
from shutil import rmtree
from datetime import timedelta
from collections import defaultdict

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel

from eval import evaluate_internal, plot_roc, accuracy, sigmoid, bootstrap, compute_cis
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score

import torch
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from conditional_dataset_train import Conditional_CTReportDataset_Train, collate_fn
from conditional_dataset_evaluate import Conditional_CTReportDataset_Eval, custom_data_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
from ct_clip import CTCLIP
import os

from scheduler import cosine_lr

from retrieval_metric import compute_ndcg, hard_retrieval_exclude_self, soft_retrieval
from datetime import datetime
import heapq

# helpers
def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

def tensor_to_nifti(tensor, path, affine=np.eye(4)):
    """
    Save tensor as a NIfTI file.

    Args:
        tensor (torch.Tensor): The input tensor with shape (D, H, W) or (C, D, H, W).
        path (str): The path to save the NIfTI file.
        affine (np.ndarray, optional): The affine matrix for the NIfTI file. Defaults to np.eye(4).
    """

    tensor = tensor.cpu()

    if tensor.dim() == 4:
        # Assume single channel data if there are multiple channels
        if tensor.size(0) != 1:
            print("Warning: Saving only the first channel of the input tensor")
        tensor = tensor.squeeze(0)
    tensor=tensor.swapaxes(0,2)
    numpy_data = tensor.detach().numpy().astype(np.float32)
    nifti_img = nib.Nifti1Image(numpy_data, affine)
    nib.save(nifti_img, path)

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1): # avg value over samples, and the num of samples, e.g. the avg loss over a batch and batchsize
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CTClipTrainer(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        num_train_steps,
        current_step,
        warmup_steps,
        local_batch_size,
        batch_size,
        data_train_npy_dir,
        data_valid_npy_dir,
        data_train_csv_dir,
        data_valid_csv_dir,
        data_train_jsonl,
        data_valid_jsonl,
        anatomy_filter,
        positive_threshold,
        negative_threshold,
        tokenizer = None,
        lr = 1.25e-6,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        train_max_samples = 30000, # take first N samples for evaluation
        valid_max_samples = 2000,
        shuffle_train_samples_every = 1000000, # if train_max_samples < total_samples, shuffle the samples every N steps
        results_folder = '/shares/menze.dqbm.uzh/ihamam/ctclip/',
        num_workers = 8,
        pin_memory = False,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        
        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            raise ValueError('Tokenzier is not defined.')

        if isinstance(num_train_steps, list):
            self.num_train_steps = num_train_steps[-1]
        else:
            self.num_train_steps = num_train_steps
            
        self.current_step = current_step
        self.batch_size = batch_size
        self.local_batch_size = local_batch_size
        
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        all_parameters = set(CTClip.parameters())

        if isinstance(lr, list):
            self.optim = get_optimizer(all_parameters, lr=lr[-1], wd=wd)
        else:
            self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        
        # Divide anatomies into different process
        
        # 这里要修改
        with open(f"/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/(all)anatomy_positive_elements_count[max_train_{int(train_max_samples)//1000}k].json", 'r') as f:
            anatomy_distribution = json.load(f)
        anatomy_samples = {anatomy:anatomy_distribution[anatomy]["train >0.9 links count"] for anatomy in anatomy_filter}   # pos 样本越多，采样越多
        
        rank = torch.distributed.get_rank()  # 获取当前进程的 rank
        world_size = torch.distributed.get_world_size()  # 获取总进程数
        distribution_train, distribution_valid = self.distribute_equal_sample_num(anatomy_samples, world_size)
        self.local_anatomy_filter_for_train, self.local_anatomy_filter_for_valid = distribution_train[rank], distribution_valid[rank]
        # for training, allow repeat when processor > anatomy 
            
        self.ds = Conditional_CTReportDataset_Train(
            local_batch_size=local_batch_size, 
            jsonl_file=data_train_jsonl, 
            csv_file_dir=data_train_csv_dir, 
            npy_file_dir=data_train_npy_dir, 
            anatomy_filter=self.local_anatomy_filter_for_train,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            max_samples=train_max_samples
            )
        
        self.valid_ds = Conditional_CTReportDataset_Eval(
            jsonl_file=data_valid_jsonl, 
            csv_file_dir=data_valid_csv_dir, 
            npy_file_dir=data_valid_npy_dir, 
            anatomy_filter=self.local_anatomy_filter_for_valid,
            max_samples=valid_max_samples
            )
        
        # self.valid_ds = Conditional_CTReportDataset_Eval(jsonl_file=data_valid_jsonl, csv_file_dir=data_valid_csv_dir, npy_file_dir=data_valid_npy_dir, anatomy_filter=anatomy_filter)
        # self.valid_subset, self.anatomy_in_this_partition = custom_data_split(self.valid_ds, rank, world_size)  # 每个process测一部分anatomy

        self.anatomy2id_ls = self.valid_ds.get_anatomy2id_ls()

        self.dl = DataLoader(
            self.ds,
            num_workers = num_workers,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            pin_memory = pin_memory,
            collate_fn = collate_fn
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers = num_workers,
            batch_size = self.batch_size * self.local_batch_size,
            shuffle = False,
            pin_memory = pin_memory,
        )
        
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        
        # check split results
        print(f"Rank {rank} anatomy subset for training: {self.local_anatomy_filter_for_train}")
        print(f"Rank {rank} total samples: {sum(anatomy_samples[k] for k in self.local_anatomy_filter_for_train)}")
        print(f"Rank {rank} anatomy subset for validation: {self.local_anatomy_filter_for_valid}")
        print(f"Rank {rank} total samples: {sum(anatomy_samples[k] for k in self.local_anatomy_filter_for_valid)}")

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTClip,
            self.optim,
        )
        
        self.lr_scheduler = cosine_lr(self.optim, lr, warmup_steps, num_train_steps)
        
        self.loss_m = AverageMeter()
        self.infoNCE_loss_m = AverageMeter()
        self.anatomy_infoNCE_loss_m = {}
        self.triplet_loss_m = AverageMeter()
        self.anatomy_triplet_loss_m = {}
        self.anatomy_valid_triplet_count_m = {}

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.shuffle_train_samples_every = shuffle_train_samples_every

        self.results_folder = Path(results_folder)
        tensorboard_path = os.path.join(results_folder, 'tensorboard_log')
        self.tb_writer = SummaryWriter(tensorboard_path)

        # if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
        #     rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)
        
        # if you want to check the 'random' performance
        # if self.is_main:
        #     self.evaluate(0)
        
    def _reset_metrics(self):
        self.loss_m.reset()
        self.triplet_loss_m.reset()
        self.infoNCE_loss_m.reset()
        for m in self.anatomy_infoNCE_loss_m.values():
            m.reset()
        for m in self.anatomy_triplet_loss_m.values():
            m.reset()
        for m in self.anatomy_valid_triplet_count_m.values():
            m.reset()

    def distribute_equal_sample_num(self, anatomy_samples, num_processes):
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
        
    def distribute_equal_anatomy(self, anatomy_samples, num_processes):
        """
        改进后的分配策略：
        1. 当anatomy数量 ≤ 进程数时：
        - 每个进程分配1个anatomy（按样本量降序）
        - 多余进程循环复用高样本量anatomy
        2. 当anatomy数量 > 进程数时：
        - 使用贪心算法均衡分配，使各进程总样本量接近，且每个进程内的anatomy数量尽可能平均
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
            # 初始化进程的anatomy数量和sample num之和
            process_anatomy_count = [0] * num_processes
            process_sample_sum = [0] * num_processes
            distribution = [[] for _ in range(num_processes)]
            
            # 分配每个anatomy
            for anatomy_name, samples in sorted_anatomy:
                # 找到当前anatomy数量最少的进程
                min_anatomy_count = min(process_anatomy_count)
                candidates = [i for i, count in enumerate(process_anatomy_count) if count == min_anatomy_count]
                
                # 在这些进程中挑选sample num之和最小的进程
                min_sample_sum = min(process_sample_sum[i] for i in candidates)
                selected_process = next(i for i in candidates if process_sample_sum[i] == min_sample_sum)
                
                # 分配anatomy到选中的进程
                distribution[selected_process].append(anatomy_name)
                process_anatomy_count[selected_process] += 1
                process_sample_sum[selected_process] += samples
            
            return distribution, distribution
        
    def evaluate(self, current_step):
        with torch.no_grad():
            
            self.CTClip.eval()
            
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
            
            fused_latents_all = {anatomy:torch.zeros(len(self.anatomy2id_ls[anatomy]), 512) for anatomy in self.local_anatomy_filter_for_valid}   
            # 每个anatomy的fused_latents顺序与dataset中的sample id顺序严格一致
            # 每个anatomy的fused_latents的shape是 (N, 512)，N是这个anatomy的样本数
            
            # First derive the fusion latent for each anatomy and each sample
            for video, anatomy_ls, sample_id_ls in tqdm(self.valid_dl, desc=f"Rank {torch.distributed.get_rank()}"):
                # video: B 1 d h w
                video = video.to(self.device).unsqueeze(1)  # B 1 1 d h w (凑出local_B)
                anatomy_ls = list(anatomy_ls)
                sample_id_ls = list(sample_id_ls)
                text_tokens=self.tokenizer(anatomy_ls, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(self.device)
                _, _, fused_latents, tmp = self.CTClip(text_tokens, video, return_latents=True, device=self.device) # B 1 Dim
                fused_latents = fused_latents.detach().cpu().squeeze()  # 去掉local_B
                for i, (sample_id, anatomy) in enumerate(zip(sample_id_ls, anatomy_ls)):
                    sample_index = self.anatomy2id_ls[anatomy].index(sample_id)
                    fused_latents_all[anatomy][sample_index] = fused_latents[i]
                     
            # Now lets calculate the similarity matrix for each anatomy
            for anatomy, fused_latents in fused_latents_all.items():
                
                fused_latents = fused_latents.to(self.device)
                image_to_image = torch.einsum('m d, n d -> m n', fused_latents, fused_latents)
                similarity_tab = self.valid_ds.get_similarity_table(anatomy)
                similarity_tab = torch.tensor(similarity_tab)
                similarity_tab = similarity_tab.to(dtype=torch.float32) * 0.01    # 0~100 uint8 -> 0~1 float32
                
                # DEBUG
                # image_to_image_np = image_to_image.cpu().numpy()
                # np.savez_compressed(f'/DB/data/haoningwu-1/zihengzhao/Ours-Testset({anatomy})-Image2Image.npz', image=image_to_image_np)
                # DEBUG
                
                assert image_to_image.shape == similarity_tab.shape, f'image_to_image {image_to_image.shape} != similarity_tab {similarity_tab.shape}'
                
                # now calculate the ndcg
                index_list = [i for i in range(similarity_tab.shape[0])]
                ndcg_scores = compute_ndcg(image_to_image, similarity_tab, index_list, k=[3, 5, 10, 20, 50, 100])
                for k, v in zip([3, 5, 10, 20, 50, 100], ndcg_scores):
                    all_metrics[f'NDCG@{k}'][anatomy] = v*100
                
                # average similarity
                pred_score, upperbound_score = soft_retrieval(image_to_image, similarity_tab, k=[3, 5, 10, 20, 50, 100])   # {'avg_similarity@{k}': xxx, ...}
                for k, v in zip([3, 5, 10, 20, 50, 100], pred_score):
                    all_metrics[f'RateScore@{k}'][anatomy] = v*100
                for k, v in zip([3, 5, 10, 20, 50, 100], upperbound_score):
                    all_metrics[f'UpperBound_RateScore@{k}'][anatomy] = v*100
                
                # hard image-image retrieval
                recall_scores = hard_retrieval_exclude_self(image_to_image, similarity_tab, index_list, k=[3, 5, 10, 20, 50, 100])
                if recall_scores == -1:
                    print(f'{anatomy} has no positive samples!')
                    for k in [3, 5, 10, 20, 50, 100]:
                        all_metrics[f'Recall@{k}'][anatomy] = -1
                else:
                    for k, v in zip([3, 5, 10, 20, 50, 100], recall_scores):
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
                    'losses': {
                        'main': (self.loss_m.sum, self.loss_m.count),
                        'triplet': (self.triplet_loss_m.sum, self.triplet_loss_m.count),
                        'infoNCE': (self.infoNCE_loss_m.sum, self.infoNCE_loss_m.count),
                        'anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m.items()},
                        'anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m.items()},
                        'anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m.items()}
                    }
                }
                
                # 执行收集（替换原 all_gather_object 调用）
                torch.distributed.all_gather_object(gathered_metrics, gathered_data)  # +++ 修改行 +++

                # On the main process, merge the dictionaries
                if self.accelerator.is_local_main_process:
                    
                    print(f'** EVAL ** Gather Done!')
                    
                    # +++ 新增：合并损失统计量 +++
                    def merge_loss(metric_name):
                        total_sum = sum(data['losses'][metric_name][0] for data in gathered_metrics)
                        total_count = sum(data['losses'][metric_name][1] for data in gathered_metrics)
                        return total_sum / total_count if total_count > 0 else 0

                    def merge_anatomy_loss(metric_name):
                        merged = defaultdict(lambda: [0, 0])  # [sum, count]
                        for data in gathered_metrics:
                            for anatomy, (s, c) in data['losses'][metric_name].items():
                                merged[anatomy][0] += s
                                merged[anatomy][1] += c
                        return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                    
                    # 计算全局平均损失
                    global_main_loss = merge_loss('main')
                    global_triplet_loss = merge_loss('triplet')
                    global_infoNCE_loss = merge_loss('infoNCE')
                    
                    # 计算解剖结构相关损失
                    global_anatomy_infoNCE = merge_anatomy_loss('anatomy_infoNCE')
                    global_anatomy_triplet = merge_anatomy_loss('anatomy_triplet')
                    global_anatomy_valid_triplet = merge_anatomy_loss('anatomy_valid_triplet')
            
                    lr = self.optim.param_groups[0]['lr']
                    
                    # +++ 修改信息输出部分 +++
                    info = f"Step {current_step} | LR {lr} | "\
                        f"Loss {global_main_loss:.4f} | "\
                        f"Triplet Loss {global_triplet_loss:.4f} | "\
                        f"InfoNCE Loss {global_infoNCE_loss:.4f} |"
                    
                    if current_step > 0:
                        
                        self.tb_writer.add_scalar('lr', lr, current_step)
                        self.tb_writer.add_scalar('train_loss', self.loss_m.avg, current_step)
                        self.tb_writer.add_scalar('infoNCE_loss', self.infoNCE_loss_m.avg, current_step)
                        self.tb_writer.add_scalar('triplet_loss', self.triplet_loss_m.avg, current_step)
                        
                        # 记录解剖结构损失
                        for anatomy, loss in global_anatomy_infoNCE.items():
                            self.tb_writer.add_scalar(f'{anatomy}_infoNCE_loss', loss, current_step)
                        for anatomy, loss in global_anatomy_triplet.items():
                            self.tb_writer.add_scalar(f'{anatomy}_triplet_loss', loss, current_step)
                        for anatomy, count in global_anatomy_valid_triplet.items():
                            self.tb_writer.add_scalar(f'{anatomy}_valid_triplet', count, current_step)
                                
                    merged_metrics = {}
                    for metric_name in all_metrics.keys():
                        merged_metrics[metric_name] = {}
                    for gathered_data in gathered_metrics:
                        for metric_name, subdict in gathered_data['metrics'].items():   # metric_name: 'Recall@5' ...
                            for key, value in subdict.items():  # pancreas: 0.8
                                merged_metrics[metric_name][key] = value
                    all_metrics = merged_metrics
                    
                    for metrics_name in ['NDCG', 'Recall', 'RateScore', 'UpperBound_RateScore']:
                        for k in [3, 5, 10, 20, 50, 100]:
                            avg_results = sum(all_metrics[f'{metrics_name}@{k}'].values()) / len(all_metrics[f'{metrics_name}@{k}'])
                            info += f' {metrics_name}@{k} {avg_results} |'
                            self.tb_writer.add_scalar(f'{metrics_name}@{k}', avg_results, current_step)
                            
                            write_info = info   # the following details will be written but not displayed
                            
                            for key, value in all_metrics[f'{metrics_name}@{k}'].items():
                                self.tb_writer.add_scalar(f'{key}_{metrics_name}@{k}', value, current_step)
                                write_info += f"{key}_{metrics_name}@{k}: {value} | "
                            
                    info += '\n'
                    write_info += '\n'
                    self.print(info)
                    with open(self.results_folder / 'log.txt', 'a') as f:
                        f.write(info)
                        
                    with open(self.results_folder / f'{current_step}.json', 'w') as f:
                        json.dump(all_metrics, f, indent=4)
                        
                    print(f'** EVAL ** Log Done!')
                    
                self._reset_metrics()

    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTClip),
            optim=self.optim.state_dict(),
            step=self.current_step
        )
        torch.save(pkg, path)

    def resume_training(self, path):
        self.print(f'** MODEL ** Resume Training from {path}')
            
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, map_location='cpu')

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        CTClip.load_state_dict(pkg['model'])
        # self.optim.load_state_dict(pkg['optim'])
        self.current_step = pkg['step'] + 1
        
    def load_checkpoint(self, path, allow_partial_load):
        self.print(f'** MODEL ** Load Checkpoint from {path}')
            
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path, map_location='cpu')

        CTClip = self.accelerator.unwrap_model(self.CTClip)
        
        if allow_partial_load:
            model_dict = CTClip.state_dict()
            pkg_state_dict = pkg['model']  # 假设 pkg['model'] 是模型的状态字典

            # 检查差异
            unexpected_state_dict = [k for k in pkg_state_dict.keys() if k not in model_dict.keys()]
            missing_state_dict = [k for k in model_dict.keys() if k not in pkg_state_dict.keys()]
            unmatchd_state_dict = [k for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape != model_dict[k].shape]

            # 加载部分参数
            state_dict = {k: v for k, v in pkg_state_dict.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
            model_dict.update(state_dict)
            CTClip.load_state_dict(model_dict)

            self.print('** MODEL ** The following parameters are UNEXPECTED in checkpoint:\n')
            self.print(unexpected_state_dict)
            self.print('** MODEL ** The following parameters are MISSING in checkpoint:\n')
            self.print(missing_state_dict)
            self.print('** MODEL ** The following parameters have DIFFERENT SHAPES in checkpoint:\n')
            self.print(unmatchd_state_dict)
            self.print('** MODEL ** The following parameters are LOADED in:\n')
            self.print(state_dict.keys())
        else:
            CTClip.load_state_dict(pkg['model'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        current_step = self.current_step
        self.lr_scheduler(current_step)

        self.CTClip.train()

        # update CTClip model
        # 有两种方案，一个是每个step同时做2D和3D的训练。另一个是每个step只做2D或3D的训练，但是要用到梯度累积。
        video, similarity_tab, anatomy, _ = next(self.dl_iter) # (B N 1 D H W), (B N N)
        video=video.to(device)
        
        similarity_tab = similarity_tab.to(dtype=torch.float32) * 0.01    # 0~100 uint8 -> 0~1 float32
        similarity_tab = similarity_tab.to(device)

        text = list(anatomy)    # a list of B str (anatomy name)
        text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        
        with self.accelerator.autocast():
            loss, triplet_loss, infoNCE_loss, valid_triplet_count = self.CTClip(text_tokens, video, gt_similarity_matrix=similarity_tab, return_loss=True, device=device)
        # loss is a scalar
        # triplet_loss infoNCE_loss are (B) shape tensor

        self.accelerator.backward(loss)
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTClip.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        
        torch.cuda.empty_cache()
        
        self.loss_m.update(loss.item(), 1)
        self.triplet_loss_m.update(triplet_loss.mean().item(), 1)
        self.infoNCE_loss_m.update(infoNCE_loss.mean().item(), 1)
        
        for i, anatomy_name in enumerate(anatomy):
            if anatomy_name not in self.anatomy_infoNCE_loss_m:
                self.anatomy_infoNCE_loss_m[anatomy_name] = AverageMeter()
                self.anatomy_triplet_loss_m[anatomy_name] = AverageMeter()
                self.anatomy_valid_triplet_count_m[anatomy_name] = AverageMeter()
            if infoNCE_loss[i].item() > 0:
                self.anatomy_infoNCE_loss_m[anatomy_name].update(infoNCE_loss[i].item(), 1)
            if triplet_loss[i].item() > 0:
                self.anatomy_triplet_loss_m[anatomy_name].update(triplet_loss[i].item(), 1)
                self.anatomy_valid_triplet_count_m[anatomy_name].update(valid_triplet_count, 1)
                
        # save model every so often
        if not (current_step % self.save_model_every) and self.is_main:
            model_path = self.results_folder / f'CTClip.{current_step}.pt'
            self.save(model_path)
            self.print(f'Saving model to {str(self.results_folder)}')
            
        if not (current_step % self.shuffle_train_samples_every):
            print(f"Rank {torch.distributed.get_rank()} | Step {current_step} | Shuffle Training Samples")
            self.ds.prepare_anatomy_data()
            # 重新创建 DataLoader
            self.dl = DataLoader(
                self.ds,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                collate_fn=collate_fn
            )
            # 重新准备迭代器
            self.dl_iter = cycle(self.dl)
            self.dl_iter = self.accelerator.prepare(self.dl_iter)
                
        # evaluate model every so often (ddp)
        if not (current_step % self.save_results_every):
            
            self.evaluate(current_step=current_step)
                        
        elif not (current_step % 100):
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                gathered_data = [{
                    'main_loss': (self.loss_m.sum, self.loss_m.count),
                    'triplet_loss': (self.triplet_loss_m.sum, self.triplet_loss_m.count),
                    'infoNCE_loss': (self.infoNCE_loss_m.sum, self.infoNCE_loss_m.count),
                    'anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m.items()},
                    'anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m.items()},
                    'anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m.items()}
                } for _ in range(world_size)]
                # 执行全局收集
                torch.distributed.all_gather_object(gathered_data, {
                    'main_loss': (self.loss_m.sum, self.loss_m.count),
                    'triplet_loss': (self.triplet_loss_m.sum, self.triplet_loss_m.count),
                    'infoNCE_loss': (self.infoNCE_loss_m.sum, self.infoNCE_loss_m.count),
                    'anatomy_infoNCE': {k: (v.sum, v.count) for k, v in self.anatomy_infoNCE_loss_m.items()},
                    'anatomy_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_triplet_loss_m.items()},
                    'anatomy_valid_triplet': {k: (v.sum, v.count) for k, v in self.anatomy_valid_triplet_count_m.items()}
                })
                
                if self.is_main:
                    # ==== 合并损失统计量 ====
                    def merge_global_loss(data_list, key):
                        total_sum = sum(d[key][0] for d in data_list)
                        total_count = sum(d[key][1] for d in data_list)
                        return total_sum / total_count if total_count > 0 else 0

                    def merge_anatomy_loss(data_list, key):
                        merged = defaultdict(lambda: [0, 0])  # [sum, count]
                        for d in data_list:
                            for anatomy, (s, c) in d[key].items():
                                merged[anatomy][0] += s
                                merged[anatomy][1] += c
                        return {k: (v[0]/v[1] if v[1]>0 else 0) for k, v in merged.items()}
                    
                    # 主进程计算全局平均
                    if torch.distributed.is_initialized():
                        global_loss = merge_global_loss(gathered_data, 'main_loss')
                        global_triplet = merge_global_loss(gathered_data, 'triplet_loss')
                        global_infoNCE = merge_global_loss(gathered_data, 'infoNCE_loss')
                        anatomy_infoNCE = merge_anatomy_loss(gathered_data, 'anatomy_infoNCE')
                        anatomy_triplet = merge_anatomy_loss(gathered_data, 'anatomy_triplet')
                        anatomy_valid_triplet = merge_anatomy_loss(gathered_data, 'anatomy_valid_triplet')
                    else:
                        # 单进程直接使用本地值
                        global_loss = self.loss_m.avg
                        global_triplet = self.triplet_loss_m.avg
                        global_infoNCE = self.infoNCE_loss_m.avg
                        anatomy_infoNCE = {k: v.avg for k, v in self.anatomy_infoNCE_loss_m.items()}
                        anatomy_triplet = {k: v.avg for k, v in self.anatomy_triplet_loss_m.items()}
                        anatomy_valid_triplet = {k: v.avg for k, v in self.anatomy_valid_triplet_count_m.items()}
                        
                    # ==== 日志记录 ====
                    lr = self.optim.param_groups[0]['lr']
                    current_time = datetime.now().strftime("%m-%d %H:%M")
                    self.print(f"Step {self.current_step} at {current_time} | LR {lr} | "
                            f"Loss {global_loss:.4f} | InfoNCE {global_infoNCE:.4f} | Triplet {global_triplet:.4f}")

                    self.tb_writer.add_scalar('lr', lr, current_step)
                    self.tb_writer.add_scalar('train_loss', global_loss, current_step)
                    self.tb_writer.add_scalar('infoNCE_loss', global_infoNCE, current_step)
                    self.tb_writer.add_scalar('triplet_loss', global_triplet, current_step)

                    # 记录解剖结构相关指标
                    for anatomy, loss in anatomy_infoNCE.items():
                        self.tb_writer.add_scalar(f'{anatomy}_infoNCE_loss', loss, current_step)
                    for anatomy, loss in anatomy_triplet.items():
                        self.tb_writer.add_scalar(f'{anatomy}_triplet_loss', loss, current_step)
                    for anatomy, count in anatomy_valid_triplet.items():
                        self.tb_writer.add_scalar(f'{anatomy}_valid_triplet', count, current_step)
                        
                self._reset_metrics()

        self.current_step += 1

    def train(self, evaluate_before_train):
        
        if evaluate_before_train:
            self.evaluate(current_step=self.current_step)   # Evaluate before training

        while self.current_step <= self.num_train_steps:
                
            self.train_step()

if __name__ == '__main__':
    anatomy_filter = ['trachea', 'bronchie', 'bone', 'heart', 'liver', 'vertebrae', 'aorta', 'pleura', 'spinal canal', 'heart ascending aorta', 'clavicle', 'breast', 'heart ventricle', 'heart atrium', 'gallbladder', 'pancreas', 'pulmonary artery', 'stomach']
    # ['esophagus', 'trachea', 'bronchie', 'bone', 'heart', 'liver', 'vertebrae', 'aorta', 'pleura'] # - 'adrenal gland'
    # ['spinal canal', 'heart ascending aorta', 'clavicle', 'breast', 'heart ventricle', 'heart atrium', 'gallbladder', 'pancreas', 'pulmonary artery', 'stomach'] # - 'thoracic vertebrae'
    # ['adrenal gland', 'aorta', 'bone', 'breast', 'bronchie', 'clavicle', 'esophagus', 'gallbladder', 'heart ascending aorta', 'heart atrium', 'heart ventricle', 'heart', 'liver', 'pancreas', 'pleura', 'pulmonary artery', 'spinal canal', 'stomach', 'thoracic vertebrae', 'vertebrae', 'trachea']
    # 最终去除了esophagus, adrenal gland和thoracic vertebrae
    
    with open("/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/(all)anatomy_positive_elements_count[max_train_12k].json", 'r') as f:
        anatomy_distribution = json.load(f)
    anatomy_samples = {anatomy:anatomy_distribution[anatomy]["train >0.9 links count"] for anatomy in anatomy_filter}   # pos 样本越多，采样越多 # {'anatomy':num, ...}
    # 将 anatomies 分配到两个 8 卡机器（共 16 个进程），保证同一机器内 8 个进程的样本总数尽量接近
    
    def distribute_equal_anatomy(anatomy_samples, num_processes):
        """
        改进后的分配策略：
        1. 当anatomy数量 ≤ 进程数时：
        - 每个进程分配1个anatomy（按样本量降序）
        - 多余进程循环复用高样本量anatomy
        2. 当anatomy数量 > 进程数时：
        - 使用贪心算法均衡分配，使各进程总样本量接近，且每个进程内的anatomy数量尽可能平均
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
            # 初始化进程的anatomy数量和sample num之和
            process_anatomy_count = [0] * num_processes
            process_sample_sum = [0] * num_processes
            distribution = [[] for _ in range(num_processes)]
            
            # 分配每个anatomy
            for anatomy_name, samples in sorted_anatomy:
                # 找到当前anatomy数量最少的进程
                min_anatomy_count = min(process_anatomy_count)
                candidates = [i for i, count in enumerate(process_anatomy_count) if count == min_anatomy_count]
                
                # 在这些进程中挑选sample num之和最小的进程
                min_sample_sum = min(process_sample_sum[i] for i in candidates)
                selected_process = next(i for i in candidates if process_sample_sum[i] == min_sample_sum)
                
                # 分配anatomy到选中的进程
                distribution[selected_process].append(anatomy_name)
                process_anatomy_count[selected_process] += 1
                process_sample_sum[selected_process] += samples
            
            return distribution, distribution
        
    distribution, _ = distribute_equal_anatomy(anatomy_samples, 8)
    
    print('Process == 8')
    
    for rank_split in distribution:
        anatomy_sample_num = [anatomy_samples[k] for k in rank_split]
        anatomy_weight = [round(math.log(num)) for num in anatomy_sample_num]   # weight 与 sqrt(size) 相关
        anatomy_weight = [w/sum(anatomy_weight) for w in anatomy_weight]  # NOTE: Balancing between anatomys
        print(rank_split, sum(anatomy_samples[k] for k in rank_split)/1000000)
        for k, w, n in zip(rank_split, anatomy_weight, anatomy_sample_num):
            print(k, w, n/1000000)
    
    # distribution, _ = distribute_anatomy(anatomy_samples, 16)
    
    # print('Process == 16')
    
    # for rank_split in distribution:
    #     print(rank_split, sum(anatomy_samples[k] for k in rank_split)/1000000)
    
    