set -e 

export MASTER_PORT=$((RANDOM % 987 + 29000))
export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # 学习率被调低了到2e-7
# 累积的梯度数一定要是（con uncon所有数据集数量的倍数）
# accelerate launch \
# --num_processes=4 \
# --num_machines=1 \
# --dynamo_backend=no \
# --mixed_precision=fp16 \
# --gradient_accumulation_steps=2 \
# --main_process_port "${MASTER_PORT}" \
# /mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
# --name 'CXR_CT_Uni_CTCXR_86' \
# --fusion_module 'crossattn' \
# --checkpoint '/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Unconditional-Image-Retrieval/CT-CLIP_v2.pt' \
# --pin_memory \
# --allow_partial_load \
# --open_vision_encoder \
# --use_triplet_loss 1.0 \
# --use_infoNCE_loss 0 \
# --use_image2image_loss 1 \
# --use_uncon_infoNCE_loss 1 \
# --use_uncon_triplet_loss 1 \
# --positive_distance_threshold 0.25 \
# --positive_threshold 0.75 \
# --stage1 \
# --anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"],["aortic valve","pericardium","pulmonary artery","aortic arch"]]' \
# --dataset_names 'CT_RATE' 'MIMIC_CXR' \
# --modality '3D' '2D' \
# --data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'  \
# --data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
# --data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity'  \
# --data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
# --data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --uncon_soft_label \
# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
# --uncon_train_filter '[["train"],["train_0","train_1","train_2","train_3"]]' \
# --uncon_valid 'test' 'test' \
# --uncon_modality '3D' '2D' \
# --save_results_every 200 \
# --save_model_every 1000 \
# --num_train_steps 5000 \
# --warmup_steps 50 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 12 120 \
# --uncon_batch_size_valid 12 120 \
# --num_workers 8

# 在只开vision encoder的情况下，batchsize可以开到8 80

accelerate launch \
--num_processes=3 \
--num_machines=1 \
--dynamo_backend=no \
--mixed_precision=fp16 \
--gradient_accumulation_steps=2 \
--main_process_port "${MASTER_PORT}" \
/mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-UNI-Image-Retrieval-CXR_CT/scripts/run_train_CXR_CT.py \
--checkpoint '//mnt/petrelfs/zhangtengfei/RadIR/CT-CLIP/CT-Conditional-Image-Retrieval-CXR_CT/log/CXR_CT_Uni_CTCXR_vision_text_aug_89/CTClip.12000.pt' \
--name 'CXR_CT_Uni_onlyCXR_vision_text_aug_812' \
--fusion_module 'crossattn' \
--pin_memory \
--allow_partial_load \
--open_pretrained_encoders \
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
--data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'  \
--data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
--data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity'  \
--data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
--data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
--data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
--uncon_soft_label \
--uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_similarity_lookup_table_valid  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
--uncon_dataset_names  'MIMIC-CXR' \
--uncon_train_filter '[["train_0","train_1","train_2","train_3"]]' \
--uncon_valid 'test' \
--uncon_modality '2D' \
--save_results_every 600 \
--save_model_every 4000 \
--num_train_steps 40000 \
--warmup_steps 1000 \
--lr 1e-6 \
--batch_size 1 2 \
--local_batch_size 10 48 \
--uncon_batch_size 80 \
--uncon_batch_size_vali 80 \
--num_workers 8

# --uncon_train_filter '[["train"],["train_0","train_1","train_2","train_3"]]' \
# --data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/train_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train.jsonl'  \
# --data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
# --data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_entity'  \
# --data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
# --data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/train_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/train_ratescore' \
# --data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --evaluate_before_train \

# --open_pretrained_encoders 

# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
# --uncon_train_filter '[["test"],["test"]]' \
# --uncon_valid 'test' 'test' \
# --uncon_modality '3D' '2D' \
# --save_results_every 100 \
# --save_model_every 1000 \
# --num_train_steps 10000 \
# --warmup_steps 50 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 10 25 \  
# --num_workers 8

# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_dataset_names 'CT_RATE' \
# --uncon_train_filter '[["test"]]' \
# --uncon_valid 'test' \
# --uncon_modality '3D'  \
# --save_results_every 100 \
# --save_model_every 1000 \
# --num_train_steps 10000 \
# --warmup_steps 50 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 10 \
# --uncon_batch_size_valid 10 \
# --num_workers 8

# 正常执行的版本
# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
# --uncon_train_filter '[["test"],["test"]]' \
# --uncon_valid 'test' 'test' \
# --uncon_modality '3D' '2D' \
# --save_results_every 20 \
# --save_model_every 1000 \
# --num_train_steps 1000 \
# --warmup_steps 10 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 6  25 \
# --uncon_batch_size_valid 6 25 \
# --num_workers 14

# --anatomy_filter  '[["pancreas","stomach","pulmonary artery","breast"],["aortic valve","pericardium","pulmonary artery","aortic arch"]]' \
# --dataset_names 'CT_RATE' 'MIMIC_CXR' \
# --modality '3D' '2D' \
# --data_train_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl'  \
# --data_valid_jsonl  '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/test_replaced.jsonl' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test.jsonl' \
# --data_train_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity'  \
# --data_valid_csv_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_entity' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_entity' \
# --data_train_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --data_valid_npy_dir '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE/anatomy/val_ratescore' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/test_ratescore' \
# --uncon_soft_label \
# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' \
# --uncon_dataset_names 'CT_RATE' \
# --uncon_train_filter '[["test"]]' \
# --uncon_valid 'test' \
# --uncon_modality '3D'  \
# --save_results_every 20 \
# --save_model_every 2000 \
# --num_train_steps 5000 \
# --warmup_steps 50 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 6 \
# --uncon_batch_size_valid 6 \
# --num_workers 8

# --uncon_similarity_lookup_table_train '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_similarity_lookup_table_valid '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_train_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_data_valid_jsonl '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CT_RATE' '/mnt/petrelfs/zhangtengfei/RadIR/dataset/CXR/MIMIC_CXR/all' \
# --uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
# --uncon_train_filter '[["train"],["train_0","train_1","train_2","train_3"]]' \
# --uncon_valid 'test' 'test' \
# --uncon_modality '3D' '2D' \
# --save_results_every 600 \
# --save_model_every 4000 \
# --num_train_steps 40000 \
# --warmup_steps 1000 \
# --lr 1e-6 \
# --batch_size 1 2 \
# --local_batch_size 10 48 \
# --uncon_batch_size 8 80 \
# --uncon_batch_size_valid 8 80 \
# --num_workers 8