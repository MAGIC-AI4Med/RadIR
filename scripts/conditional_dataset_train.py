import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
import random
import math
from monai.transforms import Compose, RandRotate, RandFlip, RandZoom, RandAffine, RandGaussianNoise, RandScaleIntensity

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

class Conditional_CTReportDataset_Train(Dataset):
    def __init__(self, local_batch_size, jsonl_file, csv_file_dir, npy_file_dir, anatomy_filter, positive_threshold, negative_threshold, max_samples=30000,modality='3D', need_aug=True):
        self.anatomy_filter = anatomy_filter
        self.csv_file_dir = csv_file_dir
        self.npy_file_dir = npy_file_dir
        self.max_samples = max_samples
        self.modality = modality
        self.local_batch_size = local_batch_size    # for each anatomy, how many samples to be selected
        
        if positive_threshold == 0 and negative_threshold != 1: # >0 & <0.4
            positive_threshold = negative_threshold
        if positive_threshold != 1 and negative_threshold == 1: # >0.7 & <1
            negative_threshold = positive_threshold
            
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        
        self.id2image_path = self.prepare_image_paths(jsonl_file)
        
        self.prepare_anatomy_data()
        
        self.anatomy_ls = list(self.anatomy2id_ls.keys())
        self.anatomy_weight = [round(math.log(len(self.anatomy2id_ls[anatomy]))) for anatomy in self.anatomy_ls]   # weight 与 sqrt(size) 相关
        self.anatomy_weight = [w/sum(self.anatomy_weight) for w in self.anatomy_weight]  # NOTE: Balancing between anatomys
        # self.anatomy_weight = [1/len(self.anatomy_ls) for anatomy in self.anatomy_ls]
        
        if need_aug:
            # input image must be CDHW
            # 修改
            if self.modality == '3D':
                self.augmentator = Compose([
                    RandRotate(range_x=45, range_y=45, range_z=45, prob=0.5, keep_size=True),  # 随机旋转
                    RandFlip(prob=0.5, spatial_axis=(0,1,2)),  # 随机翻转（沿深度轴）
                    RandZoom(min_zoom=0.5, max_zoom=1.5, prob=0.5, keep_size=True),  # 随机缩放
                    RandAffine(prob=0.5, translate_range=(32, 64, 64)),  # 随机平移
                    RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # 添加高斯噪声
                    RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5)
                ])
            elif self.modality == '2D':
                self.augmentator = Compose([
                RandRotate(range_x=45, range_y=45, prob=0.5, keep_size=True),  # 随机旋转（仅x和y轴）
                RandFlip(prob=0.5, spatial_axis=(0,1)),  # 随机翻转（仅水平和垂直轴）
                RandZoom(min_zoom=0.5, max_zoom=1.5, prob=0.5, keep_size=True),  # 随机缩放
                RandAffine(prob=0.5, translate_range=(32, 32)),  # 随机平移（仅2D平移）
                RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # 添加高斯噪声
                RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5)  # 随机调整像素强度
            ])
        else:
            self.augmentator = None
        
        if int(os.environ.get("RANK", 0)) == 0:
            print(f'** DATA ** Load {len(self.id2image_path)} images.')
            print(f'** DATA ** Load {len(self.anatomy2id_ls)} anatomy.')

    def prepare_image_paths(self, jsonl_file):
        id2image_path = {}
        
        with open(jsonl_file, 'r') as f:
            data = f.readlines()
        data = [json.loads(l) for l in data]
        for d in data:
            data_id = os.path.basename(d['img_path'])   # valid_692_a_1.nii.gz
            id2image_path[data_id] = d['img_path']

        return id2image_path
    
    def prepare_anatomy_data(self):
        self.anatomy2id_ls = {} # 'lung': ['valid_692_a_1.nii.gz', ...]
        self.anatomy2simi_tab = {} # 'lung': a tensor with shape NxN
        
        for csv_file in os.listdir(self.csv_file_dir):   
            anatomy_name = csv_file.replace('.csv', '') # lung.csv -> lung
            
            # DEBUG
            if anatomy_name not in self.anatomy_filter:
                continue
            # DEBUG
            
            self.anatomy2id_ls[anatomy_name] = []
            df = pd.read_csv(os.path.join(self.csv_file_dir, csv_file))
            print(f"{csv_file} has {len(df)} rows.")
            if len(df) > self.max_samples:
                # 随机抽取max_samples个样本，并按照原csv顺序排序，以保持与simi_tab索引对应
                sampled_df = df.sample(n=self.max_samples, random_state=42).sort_index()
                sampled_indices = sampled_df.index.tolist() # 例如抽取[0, 3, 5]行
            else:
                sampled_df = df
                sampled_indices = sampled_df.index.tolist()
            for _, row in sampled_df.iterrows():
                self.anatomy2id_ls[anatomy_name].append(row['File Path'])
                
            npy_file = csv_file.replace('.csv', '.npy')
            full_simi_tab = np.load(os.path.join(self.npy_file_dir, npy_file))
            # 截取原csv对应的大小
            full_simi_tab = full_simi_tab[:len(df), :len(df)]
            full_simi_tab = np.clip(full_simi_tab, 0, 1)
            # 根据抽样的索引构造子矩阵，确保与csv中样本一一对应
            sampled_simi_tab = full_simi_tab[np.ix_(sampled_indices, sampled_indices)] # 例如只保留[0, 3, 5]行/列
            simi_tab_scaled = np.round(sampled_simi_tab * 100).astype(np.uint8)
            self.anatomy2simi_tab[anatomy_name] = simi_tab_scaled
            
        self.check_integrity()

    def check_integrity(self):
        id_keys = set(self.anatomy2id_ls.keys())
        simi_keys = set(self.anatomy2simi_tab.keys())
        if id_keys != simi_keys:
            missing_in_simi = id_keys - simi_keys
            missing_in_id = simi_keys - id_keys
            raise ValueError(f"Inconsistent keys detected. Missing in similarity table: {missing_in_simi}. Missing in anatomy data: {missing_in_id}.")
        # 修改
        if self.modality == '3D':
            for anatomy in id_keys:
                if not (len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1]):
                    raise ValueError(f"Length of anatomy data {len(self.anatomy2id_ls[anatomy])} and similarity table {self.anatomy2simi_tab[anatomy].shape} do not match for {anatomy}.")
                
                # Here we check if all samples are readable
                filtered_sample_id_ls = []
                for i, sample_id in enumerate(self.anatomy2id_ls[anatomy]):
                    # image_file_name = sample_id.replace('.nii.gz', '.npz')
                    if sample_id in self.id2image_path and os.path.exists(self.id2image_path[sample_id]):
                        filtered_sample_id_ls.append(sample_id)
                        continue
                    else:
                        print(f'{sample_id} is missing, removed')
                        # Remove the invalid sample from self.anatomy2id_ls and update simi_tab accordingly
                        self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=0)
                        self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=1)
                self.anatomy2id_ls[anatomy] = filtered_sample_id_ls
                
                assert len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1]
        elif self.modality == '2D':
            for anatomy in id_keys:
                if not (len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1]):
                    raise ValueError(f"Length of anatomy data {len(self.anatomy2id_ls[anatomy])} and similarity table {self.anatomy2simi_tab[anatomy].shape} do not match for {anatomy}.")
                
                # Here we check if all samples are readable
                filtered_sample_id_ls = []
                for i, sample_id in enumerate(self.anatomy2id_ls[anatomy]):
                    if sample_id in self.id2image_path and os.path.exists(self.id2image_path[sample_id]):
                        filtered_sample_id_ls.append(sample_id)
                        continue
                    else:
                        print(f'{sample_id} is missing, removed')
                        # Remove the invalid sample from self.anatomy2id_ls and update simi_tab accordingly
                        self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=0)
                        self.anatomy2simi_tab[anatomy] = np.delete(self.anatomy2simi_tab[anatomy], i, axis=1)
                self.anatomy2id_ls[anatomy] = filtered_sample_id_ls
                
                assert len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[0] and len(self.anatomy2id_ls[anatomy]) == self.anatomy2simi_tab[anatomy].shape[1]
                    
    def __len__(self):
        return 10000000000
    
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
        
        # Add random shift to the crop region
        hw_max_shift = 128
        z_max_shift = 64
        if z_max_shift > 0 or hw_max_shift > 0:
            # Generate random shifts within [-max_shift, max_shift]
            h_shift = np.random.randint(-hw_max_shift, hw_max_shift + 1)
            w_shift = np.random.randint(-hw_max_shift, hw_max_shift + 1)
            d_shift = np.random.randint(-z_max_shift, z_max_shift + 1)

            # Apply shifts, ensuring the crop region remains within the image bounds
            h_start = max(h_start + h_shift, 0)
            h_end = min(h_start + dh, h)
            w_start = max(w_start + w_shift, 0)
            w_end = min(w_start + dw, w)
            d_start = max(d_start + d_shift, 0)
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
# 补充
    def load_2d_image_to_tensor(self, image_path,resize=False):
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

    def __getitem__(self, index):
        """
        Returns:
            video_tensor (tensor): N, D, H, W
            input_text (List of str): N
        """
        anatomy = np.random.choice(self.anatomy_ls, size=1, p=self.anatomy_weight)[0]
        similarity = self.anatomy2simi_tab[anatomy]
        num_samples = similarity.shape[0]
        
        # Get upper-triangular indices
        rows, cols = np.triu_indices(num_samples, k=1)
        pos_mask = (similarity[rows, cols] > self.positive_threshold) & (rows != cols)
        neg_mask = (similarity[rows, cols] < self.negative_threshold) & (rows != cols)
        pos_candidate_pairs = list(zip(rows[pos_mask], cols[pos_mask]))
        neg_candidate_pairs = list(zip(rows[neg_mask], cols[neg_mask]))
        
        # Sampling
        pos_pair_count = neg_pair_count = self.local_batch_size // 4
        if self.local_batch_size % 4 > 1:
            pos_pair_count += 1
        if len(pos_candidate_pairs) >= pos_pair_count:
            selected_pos = random.sample(pos_candidate_pairs, pos_pair_count)
        else:
            selected_pos = random.sample(list(zip(rows, cols)), pos_pair_count)
        if len(neg_candidate_pairs) >= neg_pair_count:
            selected_neg = random.sample(neg_candidate_pairs, neg_pair_count)
        else:
            selected_neg = random.sample(list(zip(rows, cols)), neg_pair_count)
        sampled_indexes = [idx for pair in (selected_pos + selected_neg) for idx in pair]
        
        # If local_batch_size is odd, add one more sample
        if self.local_batch_size % 2 == 1:
            remaining = list(set(range(num_samples)) - set(sampled_indexes))
            extra = random.choice(remaining) if remaining else random.choice(list(range(num_samples)))
            sampled_indexes.append(extra)
        
        assert len(sampled_indexes) == self.local_batch_size

        sampled_ids = [self.anatomy2id_ls[anatomy][idx] for idx in sampled_indexes]
        # sampled_ids = [sample_id for sample_id in sampled_ids if sample_id in self.id2image_path]   # WARNING 假如有一个sample_id的image不存在，那么会导致stacked_video_tensor后面的元素整体向前移动一位，和similarity_tab对不上
        
        similarity_tab = np.zeros(
            (self.local_batch_size, self.local_batch_size), 
            dtype=np.uint8  # 使用 uint8 类型
        )
                
        # 修改赋值逻辑（需要反向缩放）
        for local_i, global_i in enumerate(sampled_indexes):
            for local_j, global_j in enumerate(sampled_indexes):
                raw_value = max(
                    self.anatomy2simi_tab[anatomy][global_i, global_j],
                    self.anatomy2simi_tab[anatomy][global_j, global_i]
                )
                # 直接使用存储的 uint8 值（无需转换）
                similarity_tab[local_i, local_j] = raw_value  # 已经是 0~100 的整数
        # 修改
        if self.modality == '3D':
            stacked_video_tensor = torch.zeros((self.local_batch_size, 1, 240, 480, 480))
            
            for i, sampled_id in enumerate(sampled_ids):
                image_file_name = sampled_id.replace('.nii.gz', '.npz')
                try:
                    video_tensor = self.nii_img_to_tensor(self.id2image_path[sampled_id])
                except Exception as e:
                    print(e)
                    video_tensor = self.nii_img_to_tensor(self.id2image_path[sampled_id])

                stacked_video_tensor[i] = video_tensor
                
            if self.augmentator is not None:
                stacked_video_tensor = stacked_video_tensor.squeeze()   # N 240 480 480
                stacked_video_tensor = self.augmentator(stacked_video_tensor)
                stacked_video_tensor = stacked_video_tensor.unsqueeze(1)   # N 1 240 480 480
            
        elif self.modality == '2D':
            stacked_video_tensor = torch.zeros((self.local_batch_size, 1, 1, 480, 480))
        
            for i, sampled_id in enumerate(sampled_ids):
                video_tensor = self.load_2d_image_to_tensor(self.id2image_path[sampled_id])  # 1 1 480 480
                stacked_video_tensor[i] = video_tensor
                
            if self.augmentator is not None:
                stacked_video_tensor = stacked_video_tensor.squeeze()   # N 480 480
                stacked_video_tensor = self.augmentator(stacked_video_tensor)
                stacked_video_tensor = stacked_video_tensor.unsqueeze(1).unsqueeze(1)   # N 1 1 480 480

        return {'video_tensor': stacked_video_tensor, 'similarity_tab': torch.tensor(similarity_tab), 'anatomy': anatomy, 'sampled_ids':sampled_ids}
def collate_fn(data):
    batched_video = torch.zeros(len(data), *data[0]['video_tensor'].shape)
    batched_similarity_tab = torch.zeros(len(data), *data[0]['similarity_tab'].shape)
    batched_anatomy = []
    batched_sampled_ids = []
    
    for i in range(len(data)):
        video, similarity_tab, anatomy, sampled_ids = data[i]['video_tensor'], data[i]['similarity_tab'], data[i]['anatomy'], data[i]['sampled_ids']
        batched_video[i] = video
        batched_similarity_tab[i] = similarity_tab
        batched_anatomy.append(anatomy)
        batched_sampled_ids.append(sampled_ids)
    
    return batched_video, batched_similarity_tab, batched_anatomy, batched_sampled_ids
    
if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader, random_split
    print('** 2d **')
    dataset = Conditional_CTReportDataset_Train(
        local_batch_size=8,
        modality='2D',
        jsonl_file='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/IU_xray/all.jsonl',
        csv_file_dir='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/IU_xray/anatomy_entity',
        npy_file_dir='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/IU_xray/anatomy_ratescore',
        positive_threshold=0.75,
        negative_threshold=1,
        anatomy_filter=['left lung']
    )
    
    dl = DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle = False,
        drop_last = False,
        pin_memory = False,
        collate_fn = collate_fn
    )        
    np1 = np.load('/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/IU_xray/anatomy_ratescore/left lung.npy')
    print('相似度矩阵',np1.shape)
    dl_iter = iter(dl)
    
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    
        count = 0
        triplet_count = 0
        
        while True:
            
            count += 1
            if count > 50:
                break
            
            video, similarity_tab, anatomy, batched_sampled_ids = next(dl_iter)   # 1 8 8
            
            print(batched_sampled_ids)
            print(similarity_tab)
                    
            mask = (similarity_tab.unsqueeze(3) - similarity_tab.unsqueeze(2)) > 0.2   # N N N
            triplet_count += torch.sum(mask)
            
        print(f'** {threshold} **')
        print(similarity_tab.shape)
        print(triplet_count/count)
    
    # > 0.1 时 8x8x8 的 matrix 上平均有 _ 个有效对
    # > 0.15 时 8x8x8 的 matrix 上平均有 _ 个有效对
    # > 0.2 时 8x8x8 的 matrix 上平均有 _ 个有效对
    # > 0.25 时 8x8x8 的 matrix 上平均有 _ 个有效对
    # > 0.3 时 8x8x8 的 matrix 上平均有 _ 个有效对
    print('** CT **')
    dataset = Conditional_CTReportDataset_Train(
        local_batch_size=8,
        modality='3D',
        jsonl_file='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl',
        csv_file_dir='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity',
        npy_file_dir='/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore',
        positive_threshold=0.75,
        negative_threshold=1,
        anatomy_filter=['liver']
    )
    
    dl = DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle = False,
        drop_last = False,
        pin_memory = False,
        collate_fn = collate_fn
    )        
    
    dl_iter = iter(dl)
    
    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    
        count = 0
        triplet_count = 0
        
        while True:
            
            count += 1
            if count > 50:
                break
            
            video, similarity_tab, anatomy, batched_sampled_ids = next(dl_iter)   # 1 8 8
            
            print(batched_sampled_ids)
            print(similarity_tab)

            mask = (similarity_tab.unsqueeze(3) - similarity_tab.unsqueeze(2)) > 0.2   # N N N
            triplet_count += torch.sum(mask)
            
        print(f'** {threshold} **')
        print(similarity_tab.shape)
        print(triplet_count/count)
    