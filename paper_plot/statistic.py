import numpy as np
import pandas as pd
import os

directory = "/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/train_entity"
total_rows = 0

anatomy_counts = {}
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        anatomy_name = filename.replace('.csv', '')
        path = os.path.join(directory, filename)
        df = pd.read_csv(path)
        count = len(df)
        path = os.path.join(directory.replace('train', 'val'), filename)
        df = pd.read_csv(path)
        count += len(df)
        anatomy_counts[anatomy_name] = count

# Assuming you've collected each anatomy's count in a dictionary called anatomy_counts during the loop,
# sort the dictionary by count in descending order and print the results.
for anatomy, count in sorted(anatomy_counts.items(), key=lambda item: item[1], reverse=True):
    print(f"{anatomy}: {count}")

# Sort the dictionary by count in descending order and get the top 10 items
top10_items = sorted(anatomy_counts.items(), key=lambda item: item[1], reverse=True)[:10]
top10_anatomies = [item[0] for item in top10_items]
top10_counts = [item[1] for item in top10_items]

print(top10_anatomies, top10_counts)