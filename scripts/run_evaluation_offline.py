import argparse
from tqdm import tqdm
import pandas as pd
import os
import json
import datetime
from pathlib import Path

import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
import numpy as np

from torch.utils.data import DataLoader

from ct_clip import CTCLIP
from data import CTReportDataset
from retrieval_metric import hard_retrieval, soft_retrieval, compute_ndcg, hard_retrieval_exclude_self, hard_retrieval_exclude_self_return_details

#
# This code is to evaluate CT-CLIP with multi-grained condition
#

def prepare_anatomy_data(csv_file_dir, max_samples, anatomy_filter):
    anatomy2id_ls = {} # 'lung': ['valid_692_a_1.nii.gz', ...]
    
    for csv_file in sorted(os.listdir(csv_file_dir)):   
        anatomy_name = csv_file.replace('.csv', '') # lung.csv -> lung
        
        if anatomy_name not in anatomy_filter:
            continue
        
        anatomy2id_ls[anatomy_name] = []
        df = pd.read_csv(os.path.join(csv_file_dir, csv_file))
        for index, row in df.head(max_samples).iterrows():
            anatomy2id_ls[anatomy_name].append(row['Volumename'])
            
    return anatomy2id_ls
            
def prepare_similarity_table(npy_file_dir, max_samples, anatomy_filter):
    anatomy2simi_tab = {} # 'lung': a tensor with shape NxN
    
    for npy_file in sorted(os.listdir(npy_file_dir)):   
        anatomy_name = npy_file.replace('.npy', '') # lung.npy -> lung
        
        if anatomy_name not in anatomy_filter:
            continue
        
        simi_tab = np.load(os.path.join(npy_file_dir, npy_file))
        anatomy2simi_tab[anatomy_name] = simi_tab[:max_samples, :max_samples]
        
        for i in range(simi_tab.shape[0]):
            for j in range(simi_tab.shape[1]):
                simi_tab[i, j] = max(simi_tab[i, j], simi_tab[j, i])

    return anatomy2simi_tab

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_similarity', type=str, help='i2i similarity prediction from CT-CLIP', default='/DB/data/haoningwu-1/zihengzhao/CT-CLIP-Testset-Image2Image.npz')    # make sure the index is consistent with the data_jsonl
    parser.add_argument('--anatomy_filter', type=str, nargs='+', default=['aorta', 'liver'])
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--topk', type=int, nargs='+', default=[3, 5, 10, 20, 50, 100])
    parser.add_argument('--results_folder', type=str)    # make sure the index is consistent with the data_jsonl
    parser.add_argument('--data_jsonl', type=str, help='all images in CTRATE', default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/test.jsonl')
    parser.add_argument('--data_csv_dir', type=str, help='images under each anatomy condition', default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_entity')
    parser.add_argument('--data_npy_dir', type=str, help='i2i similarity prediction under each anatomy condition', default='/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/val_ratescore')
    parser.add_argument('--return_detail', action='store_true')
    config = parser.parse_args()
    
    results_folder = f'/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/Evaluation_Results/{config.results_folder}'
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Some preparation

    anatomy2id_ls = prepare_anatomy_data(config.data_csv_dir, config.max_samples, config.anatomy_filter)

    anatomy2simi_tab = prepare_similarity_table(config.data_npy_dir, config.max_samples, config.anatomy_filter)

    prediction_similarity = np.load(config.prediction_similarity)['image']

    with open(config.data_jsonl) as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]

    sample_id2idex = {os.path.basename(datum['img_path']):idx for idx, datum in enumerate(data)} # 'valid_692_a_1.nii.gz': 0
    
    # Start Evaluation
    
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

    for anatomy in config.anatomy_filter:
        
        similarity_tab = torch.tensor(anatomy2simi_tab[anatomy]) # NxN
        
        idx_ls = []
        id_ls = anatomy2id_ls[anatomy] # N
        filtered_id_ls = []
        for i, id_ in enumerate(id_ls):
            if id_ not in sample_id2idex:
                similarity_tab = np.delete(similarity_tab, i, axis=0)
                similarity_tab = np.delete(similarity_tab, i, axis=1)
                print(f'remove {id_}')
            else:
                idx_ls.append(sample_id2idex[id_])
                filtered_id_ls.append(id_)

        image_to_image = torch.tensor(prediction_similarity[idx_ls][:, idx_ls]) # NxN
        
        assert image_to_image.shape == similarity_tab.shape, f'{image_to_image.shape} != {similarity_tab.shape}'
        assert image_to_image.shape[0] == image_to_image.shape[1] == len(idx_ls) == len(filtered_id_ls), f'{image_to_image.shape} != {len(idx_ls)} != {len(filtered_id_ls)}'
        
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
            
            detailed_results = {}
            for query_idx, query_id in enumerate(filtered_id_ls):
                topk_results_unaligned = samplewise_topk_results[query_idx].tolist()
                topk_results = []
                for idx in topk_results_unaligned:
                    if idx >= query_idx:
                        topk_results.append(filtered_id_ls[idx+1])
                    else:
                        topk_results.append(filtered_id_ls[idx])
                # topk_results = [id_ls[idx] for row_idx, idx in enumerate(topk_results)]
                topk_scores = samplewise_topk_scores[query_idx].tolist()
                gt_topk_results_unaligned = samplewise_topk_gt_results[query_idx].tolist()
                gt_topk_results = []
                for idx in gt_topk_results_unaligned:
                    if idx >= query_idx:
                        gt_topk_results.append(filtered_id_ls[idx+1])
                    else:
                        gt_topk_results.append(filtered_id_ls[idx])
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
            
        # DEBUG
        # 抽取第一个sample
        # 保存gt的前5个结果的id和gt的相似度
        # 保存prediction的前5个结果的id和gt的相似度、prediction的相似度
        # print(all_metrics)  # 
        
        # filtered_ids = [id_ for id_ in id_ls if id_ in sample_id2idex]
        
        # similarity_tab = similarity_tab.numpy()
        # print("每个图像对应的gt前3个最相似结果:")
        # for i in range(similarity_tab.shape[0]):
        #     # 复制当前行，并将自身的相似度置为 -inf 以排除自身
        #     row_sim = similarity_tab[i].copy()
        #     row_sim[i] = -float('inf')
        #     # 返回前三个最大值的索引（从高到低排序）
        #     top3_idx = np.argsort(row_sim)[-3:][::-1]
        #     top3_ids = [filtered_ids[j] for j in top3_idx]
        #     top3_sims = [row_sim[j] for j in top3_idx]
        #     print(f"图像ID: {filtered_ids[i]}, 前3个结果: {list(zip(top3_ids, top3_sims))}")
        #     break
        
        # image_to_image = image_to_image.numpy()
        # print("每个图像预测的前3个最相似结果:")
        # for i in range(image_to_image.shape[0]):
        #     # 复制当前行，并将自身的相似度置为 -inf 以排除自身
        #     row_sim = image_to_image[i].copy()
        #     row_sim[i] = -float('inf')
        #     # 返回前三个最大值的索引（从高到低排序）
        #     top3_idx = np.argsort(row_sim)[-3:][::-1]
        #     top3_ids = [filtered_ids[j] for j in top3_idx]
        #     top3_sims = [row_sim[j] for j in top3_idx]
        #     # 对应在gt上的前三个simi
        #     top3_gt_sims = [similarity_tab[i][j] for j in top3_idx]
        #     print(f"图像ID: {filtered_ids[i]}, 前3个结果: {list(zip(top3_ids, top3_sims, top3_gt_sims))}")
        #     break
        
        # exit()
        # DEBUG
            
    # Now Print or Log
    write_info = ''
    info = ''
    for metrics_name in ['NDCG', 'Recall', 'RateScore', 'UpperBound_RateScore']:
        for k in config.topk:
            avg_results = sum(all_metrics[f'{metrics_name}@{k}'].values()) / len(all_metrics[f'{metrics_name}@{k}'])
            info += f'{metrics_name}@{k} {avg_results} | '
            write_info += f'{metrics_name}@{k} {avg_results}\n'
            
            for key, value in all_metrics[f'{metrics_name}@{k}'].items():
                write_info += f"{key}_{metrics_name}@{k}: {value} \n"
            write_info += '\n'
    print(info)
    
    with open(f'{results_folder}/log.txt', 'a') as f:
        # some basic info
        SHA_TZ = datetime.timezone(datetime.timedelta(hours=8),
                            name='Asia/Shanghai')   
        utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
        exp_time = f'{beijing_now.year}-{beijing_now.month}-{beijing_now.day}-{beijing_now.hour}-{beijing_now.minute}'
        f.write(f'{exp_time} \nConfigs :\n')
        configDict = config.__dict__
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
        # detailed results
        f.write(write_info)
        f.write('\n')
    print(f'Detailed Log at {results_folder}')
    
    with open(f'{results_folder}/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    

    
    
