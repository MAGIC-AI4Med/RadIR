set -e 
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# # export TORCH_CPP_LOG_LEVEL=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# # export NCCL_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
# # 修改你的环境变量设置
# # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:256,garbage_collection_threshold:0.6,roundup_power2_divisions:2
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8
# # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32,garbage_collection_threshold:0.9  # 最关键的代码
# export NCCL_BUFFSIZE=8388608
# export NCCL_P2P_DISABLE=1

# export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=$((RANDOM % 987 + 29000))
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 保留image latent，部分打开encoder，不加infoNCE Loss，效果如何？

# accelerate launch \
# --num_processes=4  \
# --num_machines=1 \
# --dynamo_backend=no \
# --mixed_precision=no \
# --gradient_accumulation_steps=1 \
# --main_process_port "${MASTER_PORT}" \
# /mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Conditional-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
# --name 'CXR_CT_overfit_728' \
# --fusion_module 'crossattn' \
# --open_partial_pretrained_encoders \
# --use_triplet_loss 1.0 \
# --use_infoNCE_loss 0 \
# --positive_distance_threshold 0.25 \
# --positive_threshold 0.75 \
# --anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"]]' \
# --dataset_names 'CT_RATE' \
# --modality '3D' \
# --data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' \
# --data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' \
# --data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' \
# --data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' \
# --data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' \
# --data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' \
# --save_results_every 10 \
# --save_model_every 20 \
# --num_train_steps 21 \
# --warmup_steps 2 \
# --lr 5e-7 \
# --batch_size 1 \
# --local_batch_size 5 \
# --num_workers 2
# # --evaluate_before_train \
# # --anatomy_filter  '[["bone","heart","bronchie","trachea","pleura","vertebrae","liver","aorta","spinal canal","gallbladder","clavicle","heart ascending aorta","pulmonary artery","breast","pancreas","stomach"],["lungs","heart","bones","abdomen","spine","right lung","pulmonary vasculature","aorta","airway structures","left lower lobe","right lower lobe","right upper lobe","thoracic aorta","ventricle","atrium","carina","pulmonary artery","aortic arch","bronchi","aortic valve","pericardium"]]' \
# # 下面参数要放到最前面
# # --mixed_precision=fp16 \
# # --name 'CXR_CT_overfit_v2_24final' \
# # --anatomy_filter  '[["pancreas","stomach"],["aortic valve","pericardium"]]' \

# # 学习率被调低了到2e-7

accelerate launch \
--num_processes=4  \
--num_machines=1 \
--dynamo_backend=no \
--mixed_precision=fp16 \
--gradient_accumulation_steps=2 \
--main_process_port "${MASTER_PORT}" \
/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Conditional-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
--name 'CXR_CT_overfit_two_test_729' \
--fusion_module 'crossattn' \
--open_vision_encoder \
--use_triplet_loss 1.0 \
--use_infoNCE_loss 0 \
--positive_distance_threshold 0.25 \
--positive_threshold 0.75 \
--anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"],["aortic valve","pericardium","pulmonary artery","aortic arch"]]' \
--dataset_names 'CT_RATE' 'MIMIC_CXR' \
--modality '3D' '2D' \
--data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'  \
--data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
--data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_entity'  \
--data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
--data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_ratescore' \
--data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
--save_results_every 100 \
--save_model_every 1000 \
--num_train_steps 5000 \
--warmup_steps 50 \
--lr 1e-6 \
--batch_size 1 2 \
--local_batch_size 4 24 \
--num_workers 2



# accelerate launch \
# --num_processes=4  \
# --mixed_precision=fp16 \
# --gradient_accumulation_steps=4 \
# --main_process_port "${MASTER_PORT}" \
# /mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Conditional-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
# --name 'CXR_CT_overfit_4v1' \
# --fusion_module 'crossattn' \
# --open_vision_encoder \
# --use_triplet_loss 1.0 \
# --use_infoNCE_loss 0 \
# --positive_distance_threshold 0.25 \
# --positive_threshold 0.75 \
# --anatomy_filter  '[["pancreas","stomach"],["aortic valve","pericardium"]]' \
# --dataset_names 'CT_RATE' 'MIMIC_CXR' \
# --modality '3D' '2D' \
# --data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'  \
# --data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
# --data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_entity'  \
# --data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
# --data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_ratescore' \
# --data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --save_results_every 100 \
# --save_model_every 200 \
# --num_train_steps 500 \
# --warmup_steps 5 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 4 6 \
# --num_workers 2

# --anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"],["aortic valve","pericardium","pulmonary artery","aortic arch"]]' \
# --open_partial_pretrained_encoders