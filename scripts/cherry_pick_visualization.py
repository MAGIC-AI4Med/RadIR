import json
import pandas as pd

def assign_findings_to_retrieval_results(anatomy, result_folder):

    # Step 1 read retrieval result from json
    with open(f'{result_folder}/{anatomy}.json', 'r') as f:
        result = json.load(f)
        
    # Step2 read findings from csv
    import pandas as pd

    # 读取csv文件（无表头时指定header=None，并自定义列名）
    df = pd.read_csv('/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/validation_region_report(after_filter).csv', 
                    header=None, names=['sample_id', 'anatomy_name', 'findings'])

    detail_findings = {}
    for _, row in df.iterrows():
        sample_id = row['sample_id']
        anatomy_name = row['anatomy_name']
        findings = row['findings']
        
        if sample_id not in detail_findings:
            detail_findings[sample_id] = {}
        detail_findings[sample_id][anatomy_name] = findings

    # Step 3 assign findings to retrieval result
    for sample_id, retrieval_result in result.items():
        if sample_id in detail_findings:
            retrieval_result['findings'] = detail_findings[sample_id][anatomy]
        for prediction_topk in retrieval_result['prediction_topk']:
            if prediction_topk[0] in detail_findings:
                prediction_topk.append(detail_findings[prediction_topk[0]][anatomy])
        for gt_topk in retrieval_result['gt_topk']:
            if gt_topk[0] in detail_findings:
                gt_topk.append(detail_findings[gt_topk[0]][anatomy])
        
    with open(f'{result_folder}/{anatomy}_findings.json', 'w') as f:
        json.dump(result, f, indent=4)
        
if __name__ == '__main__':
    anatomy = 'breast'
    
    result_folder = '/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/Evaluation_Results/final15_Anatomy_CT_CLIP'
    
    assign_findings_to_retrieval_results(anatomy, result_folder)
    
    result_folder = '/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/log/Evaluation_Results/final15_Anatomy_0.9_0.2_0.8_aug_sqrt(equal_anatomy_sample)_Epoch400'
    
    assign_findings_to_retrieval_results(anatomy, result_folder)
    
    # anatomy = 'liver'
    # assign_findings_to_retrieval_results(anatomy, result_folder)
    
    # anatomy = 'aorta'
    # assign_findings_to_retrieval_results(anatomy, result_folder)