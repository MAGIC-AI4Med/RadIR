set -e 

export MASTER_PORT=$((RANDOM % 987 + 29000))

accelerate launch \
--num_processes=8 \
--num_machines=1 \
--dynamo_backend=no \
--mixed_precision=fp16 \
--gradient_accumulation_steps=8 \
--main_process_port "${MASTER_PORT}" \
/scripts/run_train_CXR_CT.py \
--checkpoint "xxx.pt" \
--name 'CXRCT_UNI_stage1' \
--allow_partial_load \
--pin_memory \
--open_pretrained_encoders \
--freeze_fusion_module \
--use_image2image_loss 1 \
--use_uncon_infoNCE_loss 1 \
--use_uncon_triplet_loss 0 \
--positive_distance_threshold 0.25 \
--positive_threshold 0.75 \
--stage1 \
--uncon_soft_label \
--uncon_similarity_lookup_table_train '/dataset/CT_RATE' '/dataset/CXR/MIMIC_CXR/all' \
--uncon_similarity_lookup_table_valid '/dataset/CT_RATE' '/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_train_jsonl '/dataset/CT_RATE' '/dataset/CXR/MIMIC_CXR/all' \
--uncon_data_valid_jsonl '/dataset/CT_RATE' '/dataset/CXR/MIMIC_CXR/all' \
--uncon_dataset_names 'CT_RATE' 'MIMIC-CXR' \
--uncon_train_filter '[["train"],["train_0","train_1","train_2","train_3"]]' \
--uncon_valid 'test' 'test' \
--uncon_modality '3D' '2D' \
--train_no_aug \
--infonce_temp \
--modal_embedding \
--save_results_every 500 \
--save_model_every 6000 \
--num_train_steps 12000 \
--warmup_steps 1000 \
--lr 1e-4 \
--uncon_batch_size 6 60 \
--uncon_batch_size_valid 6 60 \
--uncon_num_workers 4

