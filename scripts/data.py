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
import tqdm
import random
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
def shuffle_sentences(text):
    """
    将一个包含多个句子的字符串拆分成单独的句子，然后随机重组这些句子。
    
    Args:
        text (str): 输入的文本字符串，包含多个由句号分隔的句子
    
    Returns:
        str: 随机重组后的句子
    """
    import random
    import re
    
    # 使用正则表达式分割句子，保留句号
    # 这种方式可以更好地处理多种句子结束情况
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    
    # 处理可能的最后一个不带句号的句子
    remaining = text[sum(len(s) for s in sentences):]
    if remaining.strip():
        sentences.append(remaining.strip() + '.')
    
    # 随机打乱句子顺序，一行代码搞定
    random.shuffle(sentences)
    
    # 直接用空格连接所有句子
    return ' '.join(s.strip() for s in sentences)
# 要提前保证数据都是存在的，且数量与对应的josnl文件长度是一致的 注意！
class CTReportDataset(Dataset):
    def __init__(self, jsonl_file,need_aug=True,modality='3D',is_train=True):
        self.modality = modality
        self.paths=[]
        self.is_train = is_train
        self.samples = self.prepare_samples(jsonl_file)
        percent = 100
        num_files = int((len(self.samples) * percent) / 100)
        #num_files = 2286
        self.samples = self.samples[:num_files]    # 是同时包含index以及对应的文本的
        print(f'** DATA ** Load {len(self.samples)} samples.')
        # 假设数据处理过了，不再需要check_integrity
        
        if need_aug: # 不行再回退到原来的版本
            # input image must be CDHW
            # 修改
            if self.modality == '3D':
               self.augmentator = Compose([
                        RandRotate(range_x=15, range_y=15, range_z=15, prob=0.3, keep_size=True),  # 随机旋转
                        RandFlip(prob=0.5, spatial_axis=0),  # 随机翻转（沿深度轴）
                        RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.3, keep_size=True),  # 随机缩放
                        RandAffine(prob=0.3, translate_range=(24, 48, 48)),  # 随机平移
                        RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),  # 添加高斯噪声
                        RandScaleIntensity(factors=[-0.1, 0.1], prob=0.3)
                ])
                # self.augmentator = Compose([
                #     # --------一次性把旋转、缩放、平移整合成 1 个 RandAffine ----------
                #     RandAffine(
                #         prob=0.75,                          # 比单独写 3 个 transform 快很多
                #         rotate_range=(np.pi/4,)*3,          # 45°
                #         translate_range=(32, 64, 64),
                #         scale_range=(0.2, 0.2, 0.2),        # 等效 RandZoom 0.8~1.2
                #         spatial_size=None,                  # keep_size=True 的含义
                #         padding_mode="border",
                #     ),
                #     # 下面这两个依然 CPU→GPU 拷贝很小，留着问题也不大
                #     RandFlip(prob=0.5, spatial_axis=(0, 1, 2)),
                #     RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
                #     RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5),
                # ])
            elif self.modality == '2D':
                self.augmentator = Compose([
                RandRotate(range_x=15, range_y=15, prob=0.5, keep_size=True),  # 随机旋转（仅x和y轴）
                RandFlip(prob=0.5, spatial_axis=(0,1)),  # 随机翻转（仅水平和垂直轴）
                RandZoom(min_zoom=0.8, max_zoom=1.1, prob=0.5, keep_size=True),  # 随机缩放
                RandAffine(prob=0.5, translate_range=(32, 32)),  # 随机平移（仅2D平移）
                RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # 添加高斯噪声
                RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5)  # 随机调整像素强度
                ])
        else:
            self.augmentator = None
        
        if int(os.environ.get("RANK", 0)) == 0:
            print(f'** DATA ** Load {len(self.samples)} images.')


    def prepare_samples(self, jsonl_file):
        samples = []
        
        with open(jsonl_file, 'r') as f:
            data = f.readlines()
        data = [json.loads(l) for l in data]
        for d in data:
            samples.append((d['img_path'], d['text']))
            self.paths.append(d['img_path'])

        return samples
        
    def __len__(self):
        return len(self.samples)
        # # 警告修改这部分
        # if self.modality == '3D':
        #     # 3D数据集
        #     return 24
        # elif self.modality == '2D':
        #     return 100
        # else:
        #     raise ValueError(f"Unsupported modality: {self.modality}. Supported modalities are '3D' and '2D'.")

    def MLS_nii_img_to_tensor(self, path):

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
        img_array = (np.array(image, dtype=np.float32) / 255.0) * 2 -1
        
        # Convert to tensor
        tensor = torch.from_numpy(img_array)
        
        # Add batch and channel dimensions
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 480, 480]
        
        return tensor

    
    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        if self.modality == '3D':
            try:
                video_tensor = self.MLS_nii_img_to_tensor(nii_file)
                assert video_tensor.dim() == 4, f"Expected 4D tensor, got {video_tensor.dim()}D tensor for {index}"
                if self.augmentator is not None and self.is_train:                    
                    video_tensor = self.augmentator(video_tensor)
                # video_tensor = torch.zeros((1, 240, 480, 480), dtype=torch.float32)
            except Exception as e:
                print(f"Error processing {nii_file} at index {index}: {e}. Skipping this sample.")
                video_tensor = torch.zeros((1, 240, 480, 480), dtype=torch.float32)
        elif self.modality == '2D':
            try:
                video_tensor = self.load_2d_image_to_tensor(nii_file, resize=True)
                video_tensor = video_tensor.squeeze().unsqueeze(0)
                if self.augmentator is not None and self.is_train:
                    video_tensor = self.augmentator(video_tensor)
                
                video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
            
            except Exception as e:
                print(f"Error processing {nii_file} at index {index}: {e}. Skipping this sample.")
                video_tensor = torch.zeros((1, 1, 480, 480), dtype=torch.float32)
            
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        
        if self.is_train:
            # 训练时打乱句子顺序
            input_text = shuffle_sentences(input_text)
        
        return video_tensor, input_text, index
 
# from accelerate import Accelerator
# import numpy as np
# import time
# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader
# from einops import rearrange

# # 初始化accelerator
# accelerator = Accelerator()
# is_main_process = accelerator.is_main_process
# # augmentator = Compose([
# #                         RandRotate(range_x=45, range_y=45, range_z=45, prob=0.5, keep_size=True),  # 随机旋转
# #                         RandFlip(prob=0.5, spatial_axis=(0,1,2)),  # 随机翻转（沿深度轴）
# #                         RandZoom(min_zoom=0.8, max_zoom=1.2, prob=0.5, keep_size=True),  # 随机缩放
# #                         RandAffine(prob=0.5, translate_range=(32, 64, 64)),  # 随机平移
# #                         RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # 添加高斯噪声
# #                         RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5)
# #                 ])


# augmentator = Compose([
#                 # --------一次性把旋转、缩放、平移整合成 1 个 RandAffine ----------
#                 RandAffine(
#                     prob=0.75,                          # 比单独写 3 个 transform 快很多
#                     rotate_range=(np.pi/4,)*3,          # 45°
#                     translate_range=(32, 64, 64),
#                     scale_range=(0.2, 0.2, 0.2),        # 等效 RandZoom 0.8~1.2
#                     spatial_size=None,                  # keep_size=True 的含义
#                     padding_mode="border",
#                 ),
#                 # 下面这两个依然 CPU→GPU 拷贝很小，留着问题也不大
#                 RandFlip(prob=0.5, spatial_axis=(0, 1, 2)),
#                 RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),
#                 RandScaleIntensity(factors=[-0.25, 0.25], prob=0.5),
#             ])
# def measure_dataloader_speed():
#     dataset_path = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test.jsonl'
#     dataset = CTReportDataset(dataset_path, need_aug=False, modality='3D', is_train=True)
#     # 速度与实际上每个gpu分配的数据集长度有关，local_len / batchsize = 2 or 4。再结合num_workers的设置
#     # 这里的batch_size是每个GPU上的batch大小，num_workers读取的情况，所以如果使用
#     # 为分布式训练配置dataloader
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=8, 
#         shuffle=True, 
#         num_workers=16,
#         pin_memory=True
#     )
    
#     # 使用accelerator准备dataloader
#     dataloader = accelerator.prepare(dataloader)
    
#     # 创建循环迭代器
#     def cycle(dl):
#         while True:
#             for data in dl:
#                 yield data
    
#     cycled_dataloader = cycle(dataloader)

#     # 在每个进程上测试加载时间
#     total_time = 0
#     times = []
#     iterations = 64
    
#     # 同步所有进程，确保公平的时间测量
#     torch.cuda.synchronize()
#     if is_main_process:
#         print(f"Testing dataloader speed (using {accelerator.num_processes} GPUs):")
    
#     for i in range(iterations):
#         # 确保所有进程同步开始
#         if dist.is_initialized():
#             dist.barrier()
        
#         start_time = time.time()
#         video, text, sample_idls = next(cycled_dataloader)
#         print(video.shape,'device:',video.device)
#         # # 将video的形状由B C D H W转为 (B C) D H W。然后再恢复
#         # B,C = video.shape[0],video.shape[1]
#         # video = rearrange(video, 'B C D H W -> (B C) D H W')
#         # video = augmentator(video)
#         # video = rearrange(video, '(B C) D H W -> B C D H W', B=B, C=C)
#         print('i', i , 'rank:', accelerator.process_index, 'sample_idls:', sample_idls)
#         # 确保GPU操作完成
#         torch.cuda.synchronize()
        
#         end_time = time.time()
#         elapsed = end_time - start_time
#         times.append(elapsed)
#         total_time += elapsed
        
#         if is_main_process:
#             print(f"Iteration {i+1}: {elapsed:.4f} seconds")

#     # 计算统计数据
#     if is_main_process:
#         avg_time = total_time / iterations
#         print(f"\nAverage loading time: {avg_time:.4f} seconds")
#         print(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s, Std: {np.std(times):.4f}s")
    
#     # 可选：收集所有进程的时间并计算全局统计信息
#     if dist.is_initialized():
#         # 收集所有进程的时间列表
#         gathered_times = [None for _ in range(dist.get_world_size())]
#         dist.all_gather_object(gathered_times, times)
        
#         # 只在主进程上处理统计
#         if is_main_process:
#             all_times = []
#             for proc_times in gathered_times:
#                 all_times.extend(proc_times)
            
#             global_avg = sum(all_times) / len(all_times)
#             print("\nStats across all processes:")
#             print(f"Global average loading time: {global_avg:.4f} seconds")
#             print(f"Global Min: {min(all_times):.4f}s, Max: {max(all_times):.4f}s, Std: {np.std(all_times):.4f}s")

# if __name__ == "__main__":
#     measure_dataloader_speed()
    
   
if __name__ == '__main__':
    # # CT部分
    # dataset = CTReportDataset(
    #     '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_filtered_replaced.jsonl', 
    #     )

    from torch.utils.data import DataLoader
    import random
    import time
    import itertools

    """测试 CTReportDataset 类的功能"""
    
    # # 初始化数据集
    # dataset_path = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all/test.jsonl'
    # dataset = CTReportDataset(dataset_path, need_aug=True, modality='2D', is_train=True)
    dataset_path = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test.jsonl'
    dataset = CTReportDataset(dataset_path, need_aug=False, modality='3D', is_train=False)
    
    print(f"数据集大小: {len(dataset)} 样本")
    
    # 测试数据集的基本信息
    print("\n1. 检查数据集基本信息...")
    print(f"数据集路径: {dataset_path}")
    print(f"文件是否存在: {os.path.exists(dataset_path)}")
    
    # 随机选择几个样本进行测试
    print("\n2. 测试随机样本...")
    # indices = random.sample(range(len(dataset)), min(5, len(dataset)))
    indices = [0, 1, 2, 3, 4]  # 测试前5个样本
    for idx in indices:
        print(f"\n样本 #{idx}:")
        try:
            # 获取样本
            video_tensor, input_text, index = dataset[idx]
            
            print(f"  - 视频张量形状: {video_tensor.shape}")
            print(f"  - 输入文本: {input_text}")
            print(f"  - 索引: {index}")
            print(f'  - 视频张量数据: {video_tensor[0, 0, 0, 0:10]}')  # 打印前10个元素
        except Exception as e:
            print(f"错误: {e}")
    
    # 测试 DataLoader
    print("\n3. 测试 DataLoader...")
    try:
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))
        
        print(f"成功加载批次数据")
        if isinstance(batch, tuple):
            for i, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    print(f"  - 批次项目 {i}: Tensor, 形状: {item.shape}")
                else:
                    print(f"  - 批次项目 {i}: {type(item)}")
        
    except Exception as e:
        print(f"DataLoader 错误: {e}")
    
    print("\n测试完成!")
    
    # dataset_path = '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test.jsonl'
    # dataset = CTReportDataset(dataset_path, need_aug=True, modality='3D', is_train=True)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=16)

    # # Create a cycle iterator
    # def cycle(dl):
    #     while True:
    #         for data in dl:
    #             yield data
    # # cycled_dataloader = itertools.cycle(dataloader)
    # cycled_dataloader = cycle(dataloader)

    # # Test loading time for 10 iterations
    # total_time = 0
    # times = []

    # print("Testing dataloader speed:")
    # for i in range(16):
    #     start_time = time.time()
    #     batch = next(cycled_dataloader)
    #     end_time = time.time()
        
    #     elapsed = end_time - start_time
    #     times.append(elapsed)
    #     total_time += elapsed
        
    #     print(f"Iteration {i+1}: {elapsed:.4f} seconds")

    # # Calculate average time
    # avg_time = total_time / 16
    # print(f"\nAverage loading time: {avg_time:.4f} seconds")
    # print(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s, Std: {np.std(times):.4f}s")
    