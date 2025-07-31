import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import random
import time

def custom_data_split(dataset, rank, world_size):
    # 计算每个进程应该处理的数据集的范围
    anatomy_split, anatomy_name = dataset.get_anatomy_split() # [[0, 10], [10, 20], ...]  # 第一个anatomy的数据对应了0～10（左闭右开）
    anatomy_split_size = [split[1] - split[0] for split in anatomy_split]
    
    sorted_indices = sorted(range(len(anatomy_split_size)), key=lambda i: anatomy_split_size[i], reverse=True)
    anatomy_split_size = [anatomy_split_size[i] for i in sorted_indices]
    anatomy_split = [anatomy_split[i] for i in sorted_indices]
    anatomy_name = [anatomy_name[i] for i in sorted_indices]
    
    anatomy_in_partitions = [[] for _ in range(world_size)]
    split_in_partitions = [[] for _ in range(world_size)]
    for i, (name, split) in enumerate(zip(anatomy_name, anatomy_split)):
        anatomy_in_partitions[i % world_size].append(name)
        split_in_partitions[i % world_size].append(split)
    
    idx_in_this_partition = []
    anatomy_in_this_partition = anatomy_in_partitions[rank]
    split_in_this_partition = split_in_partitions[rank]
    for split in split_in_this_partition:
        idx_in_this_partition += list(range(split[0], split[1]))
    
    # 使用 Subset 来选择数据集的一部分
    subset = Subset(dataset, idx_in_this_partition)

    return subset, anatomy_in_this_partition

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class Conditional_CTReportDataset_Eval(Dataset):
    def __init__(self, jsonl_file, csv_file_dir, npy_file_dir, anatomy_filter,modality='2D', max_samples=10000):
        self.anatomy_filter = anatomy_filter
        self.modality = modality
        self.id2image_path = self.prepare_image_paths(jsonl_file)
        self.id2image = {}
        self.anatomy2id_ls = self.prepare_anatomy_data(csv_file_dir, max_samples)   # DEBUG 每个类只取max_samples个测试
        self.anatomy2simi_tab = self.prepare_similarity_table(npy_file_dir, max_samples)
        self.check_integrity(self.anatomy2id_ls, self.anatomy2simi_tab)
        
        # self.anatomy_ls = list(self.anatomy2id_ls.keys()) # NOTE 不按照anatomy为单位采
        # NOTE anatomy2id_ls是形如{'lung':[a, b, c, d ...], ...}的字典，展开成id_anatomy_ls=[['lung', a], ['lung', b] ...]这样的list
        self.id_antomy_ls = [[anatomy, img_id] for anatomy, id_list in self.anatomy2id_ls.items() for img_id in id_list]
        # self.id_antomy_ls = [[anatomy, img_id] for [anatomy, img_id] in self.id_antomy_ls if img_id in self.id2image_path]    # WARNING 这样filter会导致
        
        self.anatomy_idx_range = {}
        for idx, (anatomy, _) in enumerate(self.id_antomy_ls):
            if anatomy not in self.anatomy_idx_range:
                self.anatomy_idx_range[anatomy] = [idx, idx]
            else:
                self.anatomy_idx_range[anatomy][1] = idx
        # Optionally, convert indices to tuples for clarity and print the ranges
        self.anatomy_idx_range = {anat: [r[0], r[1]] for anat, r in self.anatomy_idx_range.items()}
        
        if int(os.environ.get("RANK", 0)) == 0:
            print(f'** DATA ** Load {len(self.id2image_path)} images.')
            print(f'** DATA ** Load {len(self.anatomy2id_ls)} anatomy.')

    def prepare_image_paths(self, jsonl_file):
        id2image_path = {}
        
        with open(jsonl_file, 'r') as f:
            data = f.readlines()
        data = [json.loads(l) for l in data]
        for d in data:
            data_id = d['name']   # valid_692_a_1.nii.gz
            id2image_path[data_id] = d['img_path']

        return id2image_path
    
    def prepare_anatomy_data(self, csv_file_dir, max_samples):
        anatomy2id_ls = {} # 'lung': ['valid_692_a_1.nii.gz', ...]
        
        for csv_file in sorted(os.listdir(csv_file_dir)):   
            anatomy_name = csv_file.replace('.csv', '') # lung.csv -> lung
            
            # DEBUG
            if anatomy_name not in self.anatomy_filter:
                continue
            # DEBUG
            
            anatomy2id_ls[anatomy_name] = []
            df = pd.read_csv(os.path.join(csv_file_dir, csv_file))
            for index, row in df.head(max_samples).iterrows():
                anatomy2id_ls[anatomy_name].append(row['File Path'])

        return anatomy2id_ls
    
    def prepare_similarity_table(self, npy_file_dir, max_samples):
        anatomy2simi_tab = {} # 'lung': a tensor with shape NxN
        
        for npy_file in sorted(os.listdir(npy_file_dir)):   
            anatomy_name = npy_file.replace('.npy', '') # lung.npy -> lung
            
            # DEBUG
            if anatomy_name not in self.anatomy_filter:
                continue
            # DEBUG
            
            simi_tab = np.load(os.path.join(npy_file_dir, npy_file))[:max_samples, :max_samples]
            simi_tab = np.clip(simi_tab, 0, 1)
            simi_tab = np.round(simi_tab * 100).astype(np.uint8)  # 将 0~1 的浮点数转为 0~100 的整数 (保留两位小数)
            for i in range(simi_tab.shape[0]):
                for j in range(simi_tab.shape[0]):
                    simi_tab[i, j] = max(simi_tab[i, j], simi_tab[j, i])
            anatomy2simi_tab[anatomy_name] = simi_tab

        return anatomy2simi_tab
    
    def check_integrity(self, anatomy2id_ls, anatomy2simi_tab):
        id_keys = set(anatomy2id_ls.keys())
        simi_keys = set(anatomy2simi_tab.keys())
        if id_keys != simi_keys:
            missing_in_simi = id_keys - simi_keys
            missing_in_id = simi_keys - id_keys
            raise ValueError(f"Inconsistent keys detected. Missing in similarity table: {missing_in_simi}. Missing in anatomy data: {missing_in_id}.")
        
        for anatomy in id_keys:
            if not (len(anatomy2id_ls[anatomy]) == anatomy2simi_tab[anatomy].shape[0] and len(anatomy2id_ls[anatomy]) == anatomy2simi_tab[anatomy].shape[1]):
                raise ValueError(f"Length of anatomy data {len(anatomy2id_ls[anatomy])} and similarity table {anatomy2simi_tab[anatomy].shape} do not match for {anatomy}.")
            
            # Here we check if all samples are readable and valid (at least has one positive candidate excluding itself)
            filtered_sample_id_ls = []
            excluded_index_ls = []
            no_valid_candidate = 0
            tmp_simi_tab = anatomy2simi_tab[anatomy]
            
            for i, sample_id in enumerate(anatomy2id_ls[anatomy]):
                if np.any(np.delete(tmp_simi_tab[i], i) > 90):
                    # image_file_name = sample_id.replace('.nii.gz', '.npz')
                    if sample_id in self.id2image_path and os.path.exists(self.id2image_path[sample_id]):
                        filtered_sample_id_ls.append(sample_id)
                        continue
                    else:
                        excluded_index_ls.append(i)
                else:
                    no_valid_candidate += 1
                    excluded_index_ls.append(i)
            anatomy2id_ls[anatomy] = filtered_sample_id_ls
            
            anatomy2simi_tab[anatomy] = np.delete(anatomy2simi_tab[anatomy], excluded_index_ls, axis=0)
            anatomy2simi_tab[anatomy] = np.delete(anatomy2simi_tab[anatomy], excluded_index_ls, axis=1)
            
            print(f'For {anatomy}, {no_valid_candidate} samples has no valid candidate, removed')
            
            assert len(anatomy2id_ls[anatomy]) == anatomy2simi_tab[anatomy].shape[0] and len(anatomy2id_ls[anatomy]) == anatomy2simi_tab[anatomy].shape[1]
            
    def get_anatomy_split(self):
        anatomy_split = list(self.anatomy_idx_range.values())
        anatomy_name = list(self.anatomy_idx_range.keys())
        return anatomy_split, anatomy_name

    def __len__(self):
        # return len(self.anatomy_ls)   # NOTE 不按照anatomy为单位采
        return len(self.id_antomy_ls)
    
    def get_anatomy2id_ls(self):
        return self.anatomy2id_ls
    
    def get_similarity_table(self, anatomy):
        return self.anatomy2simi_tab[anatomy]
    
    def nii_img_to_tensor(self, path):

        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        img_data = np.flip(img_data, axis=0)
        img_data = np.flip(img_data, axis=1)

        # WARNING Respacing
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.copy(img_data)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        current = (3, 1, 1)   # this is all set to 3 1 1
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        # WARNING Normalization 2
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data ) / 1000)).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # WARNING Padding or Crop
        
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)    # h w d
        
        # Extract dimensions
        h, w, d = tensor.shape
        
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)    # d h w

        tensor = tensor.unsqueeze(0)    # 1 d h w

        return tensor
    
    def load_2d_image_to_tensor(self, image_path,resize=True):
            """
            Load a 2D grayscale image and convert it to a tensor with dimensions [1, 1, 480, 480].
            
            Args:
                image_path (str): Path to the image file.
                
            Returns:
                tensor (torch.Tensor): Tensor with shape [1, 1, 480, 480].
            """
            # Load the image as grayscale
            image = Image.open(image_path).convert('L')
            
            # Resize to 480x480 if needed
            if resize:
                if image.size != (480, 480):
                    image = image.resize((480, 480), Image.BILINEAR)
            else:
                if image.size != (480, 480):
                    # 计算可以开始裁剪的最大左上角坐标
                    width, height = image.size
                    crop_size = 480
                    if width < crop_size or height < crop_size:
                        raise ValueError(f"Image size {image.size} is smaller than the crop size {crop_size}.")
                    max_left = width - crop_size
                    max_top = height - crop_size
                    
                    # 随机选择裁剪的左上角坐标
                    left = random.randint(0, max(0, max_left))
                    top = random.randint(0, max(0, max_top))
                    
                    # 裁剪图像
                    image = image.crop((left, top, left + crop_size, top + crop_size))
                

            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Convert to tensor
            tensor = torch.from_numpy(img_array)
            
            # Add batch and channel dimensions
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 480]
            
            return tensor

    def respace_crop_pad(self, img_tensor):
        # assume we have loaded all the images
        
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        current = (3, 1, 1)   # this is all set to 3 1 1
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        
        img_data = resize_array(img_tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        # WARNING Normalization 2
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data ) / 1000)).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # WARNING Padding or Crop
        
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)    # h w d
        
        # Extract dimensions
        h, w, d = tensor.shape
        
        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)    # d h w

        tensor = tensor.unsqueeze(0)    # 1 d h w

        return tensor
    
    def load_image(self, path):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        img_data = np.flip(img_data, axis=0)
        img_data = np.flip(img_data, axis=1)

        # WARNING Respacing
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.copy(img_data)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor

    # NOTE 不按照anatomy为单位采
    # def __getitem__(self, index):
    #     """
    #     Returns:
    #         video_tensor (tensor): N, 1, D, H, W
    #         similarity_tab (tensor): N, N
    #         anatomy (str): anatomy name
    #     """
    #     anatomy = self.anatomy_ls[index]
    #     sampled_ids = self.anatomy2id_ls[anatomy]
    #     similarity_tab = self.anatomy2simi_tab[anatomy]
        
    #     stacked_video_tensor = np.zeros((len(sampled_ids), 1, 240, 480, 480))
    #     for i, sampled_id in enumerate(sampled_ids):
            
    #         # NOTE: Load and Save
    #         # print(sampled_id)
    #         # if sampled_id in self.id2image:
    #         #     image_tensor = self.id2image[sampled_id]
    #         # else:
    #         #     image_tensor = self.load_image(self.id2image_path[sampled_id])
    #         #     self.id2image[sampled_id] = image_tensor
    #         # video_tensor = self.respace_crop_pad(image_tensor)
            
    #         # NOTE: All processed in advance
    #         image_file_name = sampled_id.replace('.nii.gz', '.npz')
    #         video_tensor = np.load( f'/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/CT_CLIP_processed_image/valid/{image_file_name}')['image']
            
    #         # NOTE: Always load on the fly
    #         # image_tensor = self.load_image(self.id2image_path[sampled_id])
    #         # video_tensor = self.respace_crop_pad(image_tensor)
            
    #         stacked_video_tensor[i] = video_tensor
        
    #     return {'video_tensor': torch.tensor(stacked_video_tensor), 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy}
    
    def __getitem__(self, index):
        """
        Returns:
            video_tensor (tensor): 1, D, H, W
            anatomy (str): anatomy name
            sampled_id (str): image id
        """
        anatomy, sampled_id = self.id_antomy_ls[index]
        
        # NOTE: All processed in advance
        if self.modality == '3D':
            image_file_name = sampled_id.replace('.nii.gz', '.npz')
            if sampled_id in self.id2image_path and os.path.exists(self.id2image_path[sampled_id]):
                video_tensor = self.nii_img_to_tensor(self.id2image_path[sampled_id])
            else:
                raise ValueError(f"Image file {image_file_name} not found.")
        elif self.modality == '2D':
            if sampled_id in self.id2image_path and os.path.exists(self.id2image_path[sampled_id]):
                video_tensor = self.load_2d_image_to_tensor(self.id2image_path[sampled_id])
            else:
                raise ValueError(f"Image file {image_file_name} not found.")
        # return torch.tensor(video_tensor), anatomy, sampled_id
        return video_tensor.clone().detach(), anatomy, sampled_id
        
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from accelerate import DistributedDataParallelKwargs
    from accelerate.utils import InitProcessGroupKwargs
    
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # kwargs = InitProcessGroupKwargs()
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs])
    
    # data_valid_jsonl = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'
    # data_valid_csv_dir = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity'
    # data_valid_npy_dir = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore'
    
    # valid_ds = Conditional_CTReportDataset_Eval(jsonl_file=data_valid_jsonl, csv_file_dir=data_valid_csv_dir, npy_file_dir=data_valid_npy_dir, anatomy_filter=["left hemidiaphragm","airspace"], max_samples=10000,modality='2D')
    
    # rank = torch.distributed.get_rank()  # 获取当前进程的 rank
    # world_size = torch.distributed.get_world_size()  # 获取总进程数
    
    # subset, anatomy_in_this_partition = custom_data_split(valid_ds, rank, world_size)

    # valid_dl = DataLoader(
    #     subset,
    #     num_workers=8,
    #     batch_size=16,
    #     shuffle = False,
    # )
    
    # valid_dl = iter(valid_dl)
    
    # statistic = {}
    # for video, anatomy_ls, sampled_id_ls in iter(valid_dl):
    #     for anatomy, sample_id in zip(anatomy_ls, sampled_id_ls):
    #         if anatomy not in statistic:
    #             statistic[anatomy] = set()
    #         statistic[anatomy].add(sample_id)
            
    # print(f'RANK {rank}:', statistic.keys(), anatomy_in_this_partition)
    # for k, v in statistic.items():
    #     print(f'RANK {rank}: {k} - {len(v)}')
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs = InitProcessGroupKwargs()
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs])
    
    data_valid_jsonl = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl'
    data_valid_csv_dir = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity'
    data_valid_npy_dir = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore'
    
    valid_ds = Conditional_CTReportDataset_Eval(jsonl_file=data_valid_jsonl, csv_file_dir=data_valid_csv_dir, npy_file_dir=data_valid_npy_dir, anatomy_filter=["right rib","thyroid gland"], max_samples=10000,modality='3D')
    
    rank = torch.distributed.get_rank()  # 获取当前进程的 rank
    world_size = torch.distributed.get_world_size()  # 获取总进程数
    
    subset, anatomy_in_this_partition = custom_data_split(valid_ds, rank, world_size)

    valid_dl = DataLoader(
        subset,
        num_workers=8,
        batch_size=16,
        shuffle = False,
    )
    
    valid_dl = iter(valid_dl)
    
    statistic = {}
    for video, anatomy_ls, sampled_id_ls in iter(valid_dl):
        for anatomy, sample_id in zip(anatomy_ls, sampled_id_ls):
            if anatomy not in statistic:
                statistic[anatomy] = set()
            statistic[anatomy].add(sample_id)
            
    print(f'RANK {rank}:', statistic.keys(), anatomy_in_this_partition)
    for k, v in statistic.items():
        print(f'RANK {rank}: {k} - {len(v)}')