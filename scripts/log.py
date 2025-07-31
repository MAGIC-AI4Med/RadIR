import datetime
import os
import shutil

def move_py_files(src_path, dest_path):
    """
    递归查找src_path下所有.py文件，移动到dest_path下相同相对位置
    跳过包含特定关键词的路径

    参数：
    src_path (str): 源文件路径
    dest_path (str): 目标文件路径
    """
    # 需要排除的关键词列表（根据需求修改）
    exclude_keywords = ["/log/", "/paper_plot/", "/text_classifier/", '/data_preprocess/']  # 示例排除关键词
    
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith(".py"):
                source = os.path.join(root, file)
                
                # 检查路径是否包含排除关键词
                if any(keyword in source for keyword in exclude_keywords):
                    print(f"跳过排除文件: {source}")
                    continue
                
                rel_path = os.path.relpath(source, src_path)
                target_dir = os.path.join(dest_path, rel_path)
                
                # 优化目录创建逻辑
                os.makedirs(os.path.dirname(target_dir), exist_ok=True)
                
                target = os.path.join(target_dir, file)
                try:
                    shutil.copy(source, target)
                except Exception as e:
                    continue
                    
def log_config(f_path, config):
    # set exp time
    SHA_TZ = datetime.timezone(datetime.timedelta(hours=8),
                        name='Asia/Shanghai')   
    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)    # 北京时间
    exp_time = f'{beijing_now.year}-{beijing_now.month}-{beijing_now.day}-{beijing_now.hour}-{beijing_now.minute}'
    with open(f_path, 'a') as f:
        f.write(f'{exp_time} \n Configs :\n')
        configDict = config.__dict__
        for eachArg, value in configDict.items():
            f.write(eachArg + ' : ' + str(value) + '\n')
        f.write('\n')
    print(f'Log at {f_path}')