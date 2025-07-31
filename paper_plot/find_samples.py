import numpy as np
import pandas as pd
import os

# Start from ANATOMY 1

ANATOMY = 'lung'

simi_tab = np.load(f'/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_ratescore/{ANATOMY}.npy')

id_ls = []
df = pd.read_csv(os.path.join(f'/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_entity/{ANATOMY}.csv'))
for index, row in df.head(10000).iterrows():
    id_ls.append(row['Volumename'])

# 找到上面0.9, 0.75，0.6，0.5各一个点（不考虑对角线元素）的坐标i和j
thresholds = [0.8, 0.65, 0.5]
# Create a mask to exclude diagonal elements
mask = ~np.eye(simi_tab.shape[0], dtype=bool)
selected_idx_ls = []
selected_id_ls = []
for thr in thresholds:
    # Find indices where the similarity value matches the threshold (using a tolerance for floating point comparison)
    idx = np.argwhere((np.abs(simi_tab - thr) < 0.05) & mask)
    if idx.size > 0:
        # Take the first found coordinate and print it
        np.random.shuffle(idx)
        for candidate in idx:
            if candidate[0] in selected_idx_ls or candidate[1] in selected_idx_ls:# or np.sum(simi_tab[candidate[0], :])<simi_tab.shape[0]*0.52 or np.sum(simi_tab[candidate[1], :])<simi_tab.shape[0]*0.52:
                continue
            i, j = candidate
            break
        else:
            print(f"Threshold {thr}: No new match found.")
            continue
        print(f"Threshold {thr}: i = {i}, j = {j}")
        selected_idx_ls.append(i)
        selected_idx_ls.append(j)
        selected_id_ls.append(id_ls[i])
        selected_id_ls.append(id_ls[j])
    else:
        print(f"Threshold {thr}: No match found.")
        
# 绘制得到4个人的相互similarity
similarity_table = np.zeros((4, 4))
for local_i, i in enumerate(selected_idx_ls):
    for local_j, j in enumerate(selected_idx_ls):
        similarity_table[local_i, local_j] = np.exp(max(simi_tab[i, j], simi_tab[j, i])/0.5)

# 将对角线元素设为 -np.inf 以便在计算 softmax 时排除它们
temp = similarity_table.copy()
np.fill_diagonal(temp, -np.inf)

# 为了数值稳定性，先减去每行的最大值
max_vals = np.max(temp, axis=1, keepdims=True)
exps = np.exp(temp - max_vals)

# 确保对角线元素的 exponent 为 0
np.fill_diagonal(exps, 0)

# 计算 softmax（按行归一化）
softmax_table = exps / np.sum(exps, axis=1, keepdims=True)
print(softmax_table.tolist())

# The same patients in ANATOMY 2

ANATOMY = 'heart'

simi_tab = np.load(f'/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_ratescore/{ANATOMY}.npy')

id_ls = []
df = pd.read_csv(os.path.join(f'/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_entity/{ANATOMY}.csv'))
for index, row in df.head(10000).iterrows():
    id_ls.append(row['Volumename'])

# 找到上面0.9, 0.75，0.6，0.5各一个点（不考虑对角线元素）的坐标i和j
thresholds = [0.8, 0.65, 0.5]
# Create a mask to exclude diagonal elements
mask = ~np.eye(simi_tab.shape[0], dtype=bool)
selected_idx_ls = []
selected_id_ls = []
for thr in thresholds:
    # Find indices where the similarity value matches the threshold (using a tolerance for floating point comparison)
    idx = np.argwhere((np.abs(simi_tab - thr) < 0.05) & mask)
    if idx.size > 0:
        #  Take the first found coordinate and print it
        np.random.shuffle(idx)
        for candidate in idx:
            if candidate[0] in selected_idx_ls or candidate[1] in selected_idx_ls:
                continue
            i, j = candidate
            break
        else:
            print(f"Threshold {thr}: No new match found.")
            continue
        print(f"Threshold {thr}: i = {i}, j = {j}")
        selected_idx_ls.append(i)
        selected_idx_ls.append(j)
        selected_id_ls.append(id_ls[i])
        selected_id_ls.append(id_ls[j])
    else:
        print(f"Threshold {thr}: No match found.")
        
# 绘制得到4个人的相互similarity
similarity_table = np.zeros((4, 4))
for local_i, i in enumerate(selected_idx_ls):
    for local_j, j in enumerate(selected_idx_ls):
        similarity_table[local_i, local_j] = np.exp(max(simi_tab[i, j], simi_tab[j, i])/0.5)
        
# 将对角线元素设为 -np.inf 以便在计算 softmax 时排除它们
temp = similarity_table.copy()
np.fill_diagonal(temp, -np.inf)

# 为了数值稳定性，先减去每行的最大值
max_vals = np.max(temp, axis=1, keepdims=True)
exps = np.exp(temp - max_vals)

# 确保对角线元素的 exponent 为 0
np.fill_diagonal(exps, 0)

# 计算 softmax（按行归一化）
softmax_table = exps / np.sum(exps, axis=1, keepdims=True)
print(softmax_table.tolist())