baseline1_name = 'CT-CLIP'
baeline2_name = 'RadIR'

baseline1_result = {
    "Recall@3": {
        "esophagus": 75.23006200790405,
        "trachea": 57.429420948028564,
        "bronchie": 55.17762899398804,
        "bone": 45.75320482254028,
        "heart": 33.75104367733002,
        "liver": 72.972971200943,
        "vertebrae": 57.68667459487915,
        "aorta": 48.89867901802063,
        "pleura": 35.13513505458832
    },
    "Recall@5": {
        "esophagus": 81.21165633201599,
        "trachea": 69.24219727516174,
        "bronchie": 67.19576716423035,
        "bone": 56.33012652397156,
        "heart": 43.19131076335907,
        "liver": 77.58346796035767,
        "vertebrae": 63.68960738182068,
        "aorta": 54.03817892074585,
        "pleura": 44.594594836235046
    },
    "Recall@10": {
        "esophagus": 87.26993799209595,
        "trachea": 77.7117371559143,
        "bronchie": 75.81254839897156,
        "bone": 67.30769276618958,
        "heart": 55.722641944885254,
        "liver": 79.80921864509583,
        "vertebrae": 71.88872694969177,
        "aorta": 62.55506873130798,
        "pleura": 60.00000238418579
    }
}

baseline2_result = {
    "Recall@3": {
        "esophagus": 75.21105408668518,
        "liver": 77.97427773475647,
        "trachea": 60.178303718566895,
        "vertebrae": 63.22008967399597,
        "bronchie": 57.53217339515686,
        "aorta": 51.99999809265137,
        "bone": 49.277690052986145,
        "heart": 32.7993243932724,
        "pleura": 40.56987762451172
    },
    "Recall@5": {
        "esophagus": 81.19723796844482,
        "liver": 79.42122220993042,
        "trachea": 71.32243514060974,
        "vertebrae": 67.65140295028687,
        "bronchie": 69.19000744819641,
        "aorta": 58.5185170173645,
        "bone": 61.07544302940369,
        "heart": 44.3507581949234,
        "pleura": 54.27408218383789
    },
    "Recall@10": {
        "esophagus": 87.10667490959167,
        "liver": 80.38585186004639,
        "trachea": 80.90639114379883,
        "vertebrae": 73.26440215110779,
        "bronchie": 79.40953969955444,
        "aorta": 66.66666865348816,
        "bone": 71.26805782318115,
        "heart": 58.768969774246216,
        "pleura": 70.69199681282043
    },
}

metric_seq = ['Recall@3', 'Recall@5', 'Recall@10']
anatomy_seq = ['esophagus', 'trachea', 'bronchie', 'bone', 'heart', 'liver', 'vertebrae', 'aorta', 'pleura']

row_contents = []

# Esophagus               & 75.23 & 75.21 & 81.21 & 81.20 & 87.27 & 87.11 \\
for anatomy in anatomy_seq:
    row = f"{anatomy.capitalize():<20}"
    for metric_name in metric_seq:
        for baseline_results in [baseline1_result, baseline2_result]:
            row += f' & {baseline_results[metric_name][anatomy]:.2f}'
    row += '\\\\'
    row_contents.append(row)
    
avg_row = f"{'Average':<20}"
for metric_name in metric_seq:
    for baseline_results in [baseline1_result, baseline2_result]:
        values = [baseline_results[metric_name][anatomy] for anatomy in anatomy_seq]
        avg = sum(values) / len(values)
        avg_row += f' & {avg:.2f}'
avg_row += '\\\\'
row_contents.append(avg_row)
    
with open("/DB/data/haoningwu-1/zihengzhao/CT-Conditional-Image-Retrieval/paper_plot/conditional_table.txt", "w") as f:
    for row in row_contents:
        f.write(row + "\n")
    
