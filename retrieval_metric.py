import torch
import numpy as np

def hard_retrieval(logits, gt_logits, k):
    # treat ratescore > x as gt and calculate recall
    
    max_indices = torch.argmax(gt_logits, dim=1)
    gt_matrix = torch.zeros_like(gt_logits)
    gt_matrix[torch.arange(gt_logits.size(0)), max_indices] = 1 # for each row, the position of max element is 1, others 0
    
    # gt_matrix = torch.eye(logits.shape[0]).to(logits.device)
    
    metrics = {}    
    top10 = {}
    gt_ranks = {}

    # a sample may have multiple gt, find the one with highest logits
    gt_logit = gt_matrix * logits  # [0.1, 0.3, 0.2] * [1, 0, 1] --> [0.1, 0, 0.2]
    gt_rank = torch.argmax(gt_logit, dim=1).view(-1, 1)  # --> 2

    ranking = torch.argsort(logits, descending=True) # [0.1, 0.3, 0.2] --> [1, 2, 0] ranking logits along each row
    preds = torch.where(ranking == gt_rank)[1] # [1, 2, 0] == [0] --> [2] where the gt lies in ranks
    preds = preds.detach().cpu().numpy()
    
    if isinstance(k, int):
        return np.mean(preds < k)
    else:
        results = []
        for current_k in k:
            results.append(np.mean(preds < current_k))

    return results

def hard_retrieval_exclude_self(logits, gt_logits, index_list, k, return_detail=False):
    # treat ratescore > x as gt and calculate recall
    
    if return_detail:
        assert len(k)==1, 'must be a single k when return_detail is True'
    
    # Exclude self elements using the provided index_list
    logits = delete_elements_by_row(logits, index_list)
    gt_logits = delete_elements_by_row(gt_logits, index_list)
    logits = logits.detach().cpu()
    gt_logits = gt_logits.detach().cpu()

    # Ranking the logits in descending order for each row
    ranking = torch.argsort(logits, descending=True)

    # Define GT as all positions (excluding self) with gt_logits > 0.9
    gt_mask = gt_logits > 0.9

    # 过滤无正样本的行，只统计有GT的sample
    valid_rows = gt_mask.any(dim=1)
    if not valid_rows.any():
        return -1  # 或无正样本时的处理

    ranking = ranking[valid_rows]
    gt_mask = gt_mask[valid_rows]

    if return_detail:
        k = k[0]
        topk = ranking[:, :k]
        # Check for each row if any of the topk indices is a valid GT
        
        topk_gt = torch.gather(gt_mask, 1, topk)
        hits = topk_gt.any(dim=1).float().numpy()
        pred_topk_indices = ranking[:, :k].detach().cpu().numpy()   # we delete them selves, so need to add 1
        pred_topk_scores = logits.gather(1, ranking[:, :k]).detach().cpu().numpy()
        
        gt_topk = torch.topk(gt_logits, k, dim=1)
        gt_topk_indices = gt_topk.indices.detach().cpu().numpy()
        gt_topk_scores = gt_topk.values.detach().cpu().numpy()
        
        detail = {
            "recall": [np.mean(hits)],
            "prediction_topk_indices": pred_topk_indices,
            "prediction_topk_scores": pred_topk_scores,
            "gt_topk_indices": gt_topk_indices,
            "gt_topk_scores": gt_topk_scores,
            "row_recall": hits
        }
        return detail
    else:
        results = []
        for current_k in k:
            topk = ranking[:, :current_k]
            topk_gt = torch.gather(gt_mask, 1, topk)
            hits = topk_gt.any(dim=1).float().numpy()
            results.append(np.mean(hits))
        return results
    
def hard_retrieval_exclude_self_return_details(logits, gt_logits, index_list, k, return_detail=False):
    # treat ratescore > x as gt and calculate recall
    # to visualization, visualize retrieval details
    
    assert len(k)==1, 'must be a single k when return_detail is True'
    
    # Exclude self elements using the provided index_list
    logits = delete_elements_by_row(logits, index_list)
    gt_logits = delete_elements_by_row(gt_logits, index_list)
    logits = logits.detach().cpu()
    gt_logits = gt_logits.detach().cpu()

    # Ranking the logits in descending order for each row
    ranking = torch.argsort(logits, descending=True)

    # Define GT as all positions (excluding self) with gt_logits > 0.9
    gt_mask = gt_logits > 0.9

    k = k[0]
    topk = ranking[:, :k]
    # Check for each row if any of the topk indices is a valid GT
    
    topk_gt = torch.gather(gt_mask, 1, topk)
    hits = topk_gt.any(dim=1).float().numpy()
    pred_topk_indices = ranking[:, :k].detach().cpu().numpy()   # we delete them selves, so need to add 1
    pred_topk_scores = gt_logits.gather(1, ranking[:, :k]).detach().cpu().numpy()
    
    gt_topk = torch.topk(gt_logits, k, dim=1)
    gt_topk_indices = gt_topk.indices.detach().cpu().numpy()
    gt_topk_scores = gt_topk.values.detach().cpu().numpy()
    
    detail = {
        "recall": [np.mean(hits)],
        "prediction_topk_indices": pred_topk_indices,
        "prediction_topk_scores": pred_topk_scores,
        "gt_topk_indices": gt_topk_indices,
        "gt_topk_scores": gt_topk_scores,
        "row_recall": hits
    }
    return detail

def soft_retrieval(logits, similarity_matrix, k=[1, 5, 10]):
    # calculate avg and upperbound ratescore of image 2 image retrieval results
    # NOTE: do not have to exclude items themselves in logits or similarity_matrix
    
    ranking = torch.argsort(logits, descending=True) # [0.1, 0.3, 0.2] --> [1, 2, 0] ranking logits along each row
    pred_score = []
    upperbound_score = []

    for current_k in k:
        top_ranking = ranking[:, 1:current_k+1] # [n, k] exclude themselves (top 1)
        avg_similarities = torch.zeros(top_ranking.shape[0])
        for i in range(top_ranking.shape[0]):
            top_idx = top_ranking[i].tolist()
            avg_similarities[i] = similarity_matrix[i, top_idx].mean()
        pred_score.append(avg_similarities.mean().item())

    for current_k in k:
        actual_k = min(current_k + 1, similarity_matrix.size(1))
        topk_values = torch.topk(similarity_matrix, actual_k, dim=1).values  # shape: (n_samples, actual_k)
        # 如果实际取出的topk不足以排除 self，则直接使用所有
        if actual_k > 1:
            topk_without_self = topk_values[:, 1:]
        else:
            topk_without_self = topk_values
        upperbound_score.append(topk_without_self.mean().item())

    return pred_score, upperbound_score

def soft_retrieval_uncon(logits, gt_logits, k):
    # treat ratescore > x as gt and calculate recall
    
    logits = logits.detach().cpu()
    gt_logits = gt_logits.detach().cpu()

    # Ranking the logits in descending order for each row
    ranking = torch.argsort(logits, descending=True)

    # Define GT as all positions (excluding self) with gt_logits > 0.9
    gt_mask = gt_logits > 0.9

    # 过滤无正样本的行，只统计有GT的sample
    valid_rows = gt_mask.any(dim=1)
    if not valid_rows.any():
        return -1  # 或无正样本时的处理

    ranking = ranking[valid_rows]
    gt_mask = gt_mask[valid_rows]

    if isinstance(k, int):
        topk = ranking[:, :k]
        # Check for each row if any of the topk indices is a valid GT
        topk_gt = torch.gather(gt_mask, 1, topk)
        hits = topk_gt.any(dim=1).float().numpy()
        return np.mean(hits)
    else:
        results = []
        for current_k in k:
            topk = ranking[:, :current_k]
            topk_gt = torch.gather(gt_mask, 1, topk)
            hits = topk_gt.any(dim=1).float().numpy()
            results.append(np.mean(hits))
        return results

def soft_retrieval_exclude_self(logits, gt_logits, index_list, k):
    # treat ratescore > x as gt and calculate recall
    logits = delete_elements_by_row(logits, index_list)
    gt_logits = delete_elements_by_row(gt_logits, index_list)
    logits = logits.detach().cpu()
    gt_logits = gt_logits.detach().cpu()

    # Ranking the logits in descending order for each row
    ranking = torch.argsort(logits, descending=True)

    # Define GT as all positions (excluding self) with gt_logits > 0.9
    gt_mask = gt_logits > 0.9

    # 过滤无正样本的行，只统计有GT的sample
    valid_rows = gt_mask.any(dim=1)
    if not valid_rows.any():
        return -1  # 或无正样本时的处理
    ranking = ranking[valid_rows]
    gt_mask = gt_mask[valid_rows]

    if isinstance(k, int):
        topk = ranking[:, :k]
        # Check for each row if any of the topk indices is a valid GT
        topk_gt = torch.gather(gt_mask, 1, topk)
        hits = topk_gt.any(dim=1).float().numpy()
        return np.mean(hits)
    else:
        results = []
        for current_k in k:
            topk = ranking[:, :current_k]
            topk_gt = torch.gather(gt_mask, 1, topk)
            hits = topk_gt.any(dim=1).float().numpy()
            results.append(np.mean(hits))
        print('results:', results)
        return results


def RateScore_retrieval(logits, similarity_matrix, k=[1, 5, 10]):
    # calculate avg and upperbound ratescore of retrieval
    # NOTE: do not have to exclude items themselves in logits or similarity_matrix
    
    ranking = torch.argsort(logits, descending=True) # [0.1, 0.3, 0.2] --> [1, 2, 0] ranking logits along each row
    pred_score = []
    upperbound_score = []

    for current_k in k:
        top_ranking = ranking[:, 1:current_k+1] # [n, k] exclude themselves (top 1)
        avg_similarities = torch.zeros(top_ranking.shape[0])
        for i in range(top_ranking.shape[0]):
            top_idx = top_ranking[i].tolist()
            avg_similarities[i] = similarity_matrix[i, top_idx].mean()
        pred_score.append(avg_similarities.mean().item())

    for current_k in k:
        actual_k = min(current_k + 1, similarity_matrix.size(1))
        topk_values = torch.topk(similarity_matrix, actual_k, dim=1).values  # shape: (n_samples, actual_k)
        # 如果实际取出的topk不足以排除 self，则直接使用所有
        if actual_k > 1:
            topk_without_self = topk_values[:, 1:]
        else:
            topk_without_self = topk_values
        upperbound_score.append(topk_without_self.mean().item())

    return pred_score, upperbound_score


def compute_ndcg_exclude_self(pred_scores, true_scores, index_list, k=None):
    """
    计算NDCG。
    当 k 是一个整数时，计算对应 k 值的 ndcg；当 k 是列表时，分别计算每个 k 的 ndcg，返回一个与 k 列表等长的列表，每个元素是对应的 score。
    输入的 true_scores 和 pred_scores 为 tensor 类型。
    :param true_scores: 真实相似度矩阵 (n_query, n_samples)
    :param pred_scores: 模型预测的相似度矩阵 (n_query, n_samples)
    :param index_list: 每行需要删除的元素的索引列表
    :param k: 只考虑前 k 个结果。如果为 None，则考虑所有结果；如果为 list，则分别计算每个 k 的 ndcg
    :return: 当 k 为 list 时返回一个 ndcg 值的列表，否则返回单个 ndcg 值
    """
    true_scores = delete_elements_by_row(true_scores, index_list)
    pred_scores = delete_elements_by_row(pred_scores, index_list)
    # 转换为 numpy 数组以便后续排序和计算
    true_scores = true_scores.detach().cpu().numpy()
    pred_scores = pred_scores.detach().cpu().numpy()
    n_query = true_scores.shape[0]

    if k is None:
        k_list = [true_scores.shape[1]]
    elif isinstance(k, list):
        k_list = k
    else:
        k_list = [k]

    ndcg_results = []
    for current_k in k_list:
        ndcg_scores = []
        for i in range(n_query):
            true = true_scores[i]
            pred = pred_scores[i]
            pred_rank = np.argsort(pred)[::-1]
            true_sorted_by_pred = true[pred_rank]

            # 计算 DCG
            if current_k is not None:
                true_sorted_by_pred_k = true_sorted_by_pred[:current_k]
            else:
                true_sorted_by_pred_k = true_sorted_by_pred
            dcg = compute_dcg(true_sorted_by_pred_k)

            # 计算 IDCG
            ideal_sorted_true = np.sort(true)[::-1]
            if current_k is not None:
                ideal_sorted_true = ideal_sorted_true[:current_k]
            idcg = compute_dcg(ideal_sorted_true)

            ndcg = dcg / idcg if idcg != 0 else 0
            ndcg_scores.append(ndcg)
        ndcg_results.append(np.mean(ndcg_scores))

    # 如果原始 k 不是 list，则返回单个值
    if not isinstance(k, list):
        return ndcg_results[0]
    return ndcg_results

def compute_ndcg_uncon(pred_scores, true_scores, k=None):
    """
    计算NDCG。
    当 k 是一个整数时，计算对应 k 值的 ndcg；当 k 是列表时，分别计算每个 k 的 ndcg，返回一个与 k 列表等长的列表，每个元素是对应的 score。
    输入的 true_scores 和 pred_scores 为 tensor 类型。
    :param true_scores: 真实相似度矩阵 (n_query, n_samples)
    :param pred_scores: 模型预测的相似度矩阵 (n_query, n_samples)
    :param k: 只考虑前 k 个结果。如果为 None，则考虑所有结果；如果为 list，则分别计算每个 k 的 ndcg
    :return: 当 k 为 list 时返回一个 ndcg 值的列表，否则返回单个 ndcg 值
    """
    # 转换为 numpy 数组以便后续排序和计算
    true_scores = true_scores.detach().cpu().numpy()
    pred_scores = pred_scores.detach().cpu().numpy()
    n_query = true_scores.shape[0]

    if k is None:
        k_list = [true_scores.shape[1]]
    elif isinstance(k, list):
        k_list = k
    else:
        k_list = [k]

    ndcg_results = []
    for current_k in k_list:
        ndcg_scores = []
        for i in range(n_query):
            true = true_scores[i]
            pred = pred_scores[i]
            pred_rank = np.argsort(pred)[::-1]
            true_sorted_by_pred = true[pred_rank]

            # 计算 DCG
            if current_k is not None:
                true_sorted_by_pred_k = true_sorted_by_pred[:current_k]
            else:
                true_sorted_by_pred_k = true_sorted_by_pred
            dcg = compute_dcg(true_sorted_by_pred_k)

            # 计算 IDCG
            ideal_sorted_true = np.sort(true)[::-1]
            if current_k is not None:
                ideal_sorted_true = ideal_sorted_true[:current_k]
            idcg = compute_dcg(ideal_sorted_true)

            ndcg = dcg / idcg if idcg != 0 else 0
            ndcg_scores.append(ndcg)
        ndcg_results.append(np.mean(ndcg_scores))

    # 如果原始 k 不是 list，则返回单个值
    if not isinstance(k, list):
        return ndcg_results[0]
    return ndcg_results


def delete_elements_by_row(matrix, delete_list):
    """
    根据给定的索引列表，删除每行中指定位置的元素
    :param matrix: 输入的 n*m 维 tensor
    :param delete_list: 一个 n 大小的 list，表示每行需要删除的元素的索引
    :return: 处理后的 n*(m-1) 维 tensor
    """
    n, m = matrix.size()
    result = []

    for i in range(n):
        row = matrix[i]
        delete_index = delete_list[i]
        # 删除指定位置的元素
        new_row = torch.cat([row[:delete_index], row[delete_index+1:]])
        result.append(new_row)

    return torch.stack(result)

def compute_dcg(scores):
    """计算DCG"""
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

def compute_ndcg(pred_scores, true_scores, index_list, k=None):
    """
    计算NDCG。
    当 k 是一个整数时，计算对应 k 值的 ndcg；当 k 是列表时，分别计算每个 k 的 ndcg，返回一个与 k 列表等长的列表，每个元素是对应的 score。
    输入的 true_scores 和 pred_scores 为 tensor 类型。
    :param true_scores: 真实相似度矩阵 (n_query, n_samples)
    :param pred_scores: 模型预测的相似度矩阵 (n_query, n_samples)
    :param index_list: 每行需要删除的元素的索引列表
    :param k: 只考虑前 k 个结果。如果为 None，则考虑所有结果；如果为 list，则分别计算每个 k 的 ndcg
    :return: 当 k 为 list 时返回一个 ndcg 值的列表，否则返回单个 ndcg 值
    """
    true_scores = delete_elements_by_row(true_scores, index_list)
    pred_scores = delete_elements_by_row(pred_scores, index_list)
    # 转换为 numpy 数组以便后续排序和计算
    true_scores = true_scores.detach().cpu().numpy()
    pred_scores = pred_scores.detach().cpu().numpy()
    n_query = true_scores.shape[0]

    if isinstance(k, list):
        k_list = k
    else:
        k_list = [k]

    ndcg_results = []
    for current_k in k_list:
        ndcg_scores = []
        for i in range(n_query):
            true = true_scores[i]
            pred = pred_scores[i]
            pred_rank = np.argsort(pred)[::-1]
            true_sorted_by_pred = true[pred_rank]

            # 计算 DCG
            if current_k is not None:
                true_sorted_by_pred_k = true_sorted_by_pred[:current_k]
            else:
                true_sorted_by_pred_k = true_sorted_by_pred
            dcg = compute_dcg(true_sorted_by_pred_k)

            # 计算 IDCG
            ideal_sorted_true = np.sort(true)[::-1]
            if current_k is not None:
                ideal_sorted_true = ideal_sorted_true[:current_k]
            idcg = compute_dcg(ideal_sorted_true)

            ndcg = dcg / idcg if idcg != 0 else 0
            ndcg_scores.append(ndcg)
        ndcg_results.append(np.mean(ndcg_scores))

    # 如果原始 k 不是 list，则返回单个值
    if not isinstance(k, list):
        return ndcg_results[0]
    return ndcg_results



if __name__ == '__main__':
    logits = torch.rand(4, 4)
    logits.fill_diagonal_(1)
    gt_logits = torch.rand(4, 4)
    gt_logits.fill_diagonal_(1)
    index = [0, 1, 2, 3]
    
    print('logits:', logits)
    print('gt_logits:', gt_logits)
    
    ndcg = compute_ndcg(gt_logits, logits, index, k=[1, 3])
    
    print(ndcg)