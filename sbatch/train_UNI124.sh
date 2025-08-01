set -e 

export MASTER_PORT=$((RANDOM % 987 + 29000))
export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # 学习率被调低了到2e-7
# 累积的梯度数一定要是（con uncon所有数据集数量的倍数）
accelerate launch \
--num_processes=2  \
--num_machines=1 \
--dynamo_backend=no \
--mixed_precision=fp16 \
--gradient_accumulation_steps=4 \
--main_process_port "${MASTER_PORT}" \
/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
--checkpoint '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Unconditional-Image-Retrieval/CT-CLIP_v2.pt' \
--name 'CXR_CT_Uni' \
--fusion_module 'crossattn' \
--open_vision_encoder \
--use_triplet_loss 1.0 \
--use_infoNCE_loss 0 \
--use_image2image_loss 1 \
--use_uncon_infoNCE_loss 1 \
--use_uncon_triplet_loss 1 \
--positive_distance_threshold 0.25 \
--positive_threshold 0.75 \
--stage1 \
--anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"],["aortic valve","pericardium","pulmonary artery","aortic arch"]]' \
--dataset_names 'CT_RATE' 'MIMIC_CXR' \
--modality '3D' '2D' \
--data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'  \
--data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
--data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_entity'  \
--data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
--data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_ratescore' \
--data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
--uncon_soft_label \
--uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
--uncon_train_filter '[["test"],["test"]]' \
--uncon_valid 'test' 'test' \
--uncon_modality '3D' '2D' \
--save_results_every 100 \
--save_model_every 1000 \
--num_train_steps 5000 \
--warmup_steps 50 \
--lr 1e-6 \
--batch_size 1 2 \
--local_batch_size 10 48 \
--uncon_batch_size 5 10 \
--num_workers 2

# --uncon_train_filter '[["train"],["train_0","train_1","train_2","train_3"]]' \