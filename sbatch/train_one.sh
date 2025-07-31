set -e 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DIST_DEBUG=DETAIL            # 打印 NCCL/CUDA 详细堆栈
# export TORCH_CPP_LOG_LEVEL=INFO
export TORCHELASTIC_ERROR_FILE=/mnt/petrelfs/zhangtengfei/RadIR/error.json
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG=INFO

export CUDA_LAUNCH_BLOCKING=1
export MASTER_PORT=$((RANDOM % 1000 + 29000))
export CUDA_HOME=/mnt/petrelfs/share/cuda-12.0/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 保留image latent，部分打开encoder，不加infoNCE Loss，效果如何？

accelerate launch \
--mixed_precision=fp16 \
--num_processes=2  \
--main_process_port "${MASTER_PORT}" \
/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Conditional-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT_one.py \
--name 'CXR_CT_overfit_v2_21' \
--fusion_module 'crossattn' \
--open_vision_encoder \
--use_triplet_loss 1.0 \
--use_infoNCE_loss 0 \
--positive_distance_threshold 0.25 \
--positive_threshold 0.75 \
--anatomy_filter  '[["pancreas","stomach"],["aortic valve","pericardium"]]' \
--dataset_names 'CT_RATE' 'MIMIC_CXR' \
--modality '3D' '2D' \
--data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'  \
--data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
--data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_entity'  \
--data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
--data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_ratescore' \
--data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
--save_results_every 10 \
--save_model_every 10 \
--num_train_steps 50 \
--warmup_steps 5 \
--lr 1e-6 \
--batch_size 1 2 \
--local_batch_size 8 24 \
--num_workers 2
# --evaluate_before_train \
# --anatomy_filter  '[["bone","heart","bronchie","trachea","pleura","vertebrae","liver","aorta","spinal canal","gallbladder","clavicle","heart ascending aorta","pulmonary artery","breast","pancreas","stomach"],["lungs","heart","bones","abdomen","spine","right lung","pulmonary vasculature","aorta","airway structures","left lower lobe","right lower lobe","right upper lobe","thoracic aorta","ventricle","atrium","carina","pulmonary artery","aortic arch","bronchi","aortic valve","pericardium"]]' \
# 下面参数要放到最前面
# --mixed_precision=fp16 \