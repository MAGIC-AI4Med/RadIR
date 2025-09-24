set -e 

export MASTER_PORT=$((RANDOM % 987 + 29000))

accelerate launch \
--num_processes=4 \
--num_machines=1 \
--dynamo_backend=no \
--mixed_precision=fp16 \
--gradient_accumulation_steps=8 \
--main_process_port "${MASTER_PORT}" \
/scripts/run_CXR_CT_train.py \
--checkpoint "xxx" \
--name 'CXRCT_UNI_stage1_and_stage2' \
--allow_partial_load \
--pin_memory \
--open_pretrained_encoders \
--open_fusion_module \
--use_triplet_loss 1.0 \
--use_infoNCE_loss 0.0 \
--use_image2image_loss 1 \
--use_uncon_infoNCE_loss 1 \
--use_uncon_triplet_loss 0 \
--positive_distance_threshold 0.25 \
--positive_threshold 0.75 \
--stage1 \
--stage2 \
--anatomy_filter  '[["bone","heart","bronchie","trachea","pleura","vertebrae","liver","aorta","spinal canal","gallbladder","clavicle","heart ascending aorta","pulmonary artery","breast","pancreas","stomach"],["lungs","heart","bones","abdomen","spine","right lung","pulmonary vasculature","aorta","airway structures","left lower lobe","right lower lobe","right upper lobe","thoracic aorta","ventricle","atrium","carina","pulmonary artery","aortic arch","bronchi","costophrenic angle","aortic valve","pericardium"]]' \
--dataset_names 'CT_RATE' 'MIMIC_CXR' \
--modality '3D' '2D' \
--data_train_jsonl  '/dataset/CT_RATE/train.jsonl' '/dataset/CXR/MIMIC_CXR/train.jsonl'  \
--data_valid_jsonl  '/dataset/CT_RATE/test.jsonl' '/dataset/CXR/MIMIC_CXR/test.jsonl' \
--data_train_csv_dir '/dataset/CT_RATE/anatomy/train_entity' '/dataset/CXR/MIMIC_CXR/train_entity'  \
--data_valid_csv_dir '/dataset/CT_RATE/anatomy/val_entity' '/dataset/CXR/MIMIC_CXR/test_entity' \
--data_train_npy_dir '/dataset/CT_RATE/anatomy/train_ratescore' '/dataset/CXR/MIMIC_CXR/train_ratescore' \
--data_valid_npy_dir '/dataset/CT_RATE/anatomy/val_ratescore' '/dataset/CXR/MIMIC_CXR/test_ratescore' \
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
--batch_size 1 2 \
--local_batch_size 6 30 \
--uncon_batch_size 6 60 \
--uncon_batch_size_valid 6 60 \
--uncon_num_workers 4 \
--con_num_workers 4


