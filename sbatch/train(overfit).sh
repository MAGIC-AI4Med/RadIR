set -e 
export MASTER_PORT=$((RANDOM % 987 + 29000))
export CUDA_HOME=/mnt/petrelfs/share/cuda-12.4/
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
--num_processes=7 \
--mixed_precision=fp16 \
--main_process_port 29501 \
/DB/data/haoningwu-1/zihengzhao/CT-Unconditional-Image-Retrieval/scripts/run_train.py \
--name 'triple_loss(overfit)' \
--checkpoint '/DB/data/haoningwu-1/zihengzhao/CT-CLIP_v2_wrapped.pt' \
--data_train_jsonl '/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/test.jsonl' \
--data_valid_jsonl '/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/test.jsonl' \
--similarity_lookup_table_train '/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/CT_test_ratescore.npy' \
--similarity_lookup_table_valid '/DB/data/haoningwu-1/zihengzhao/data/tengfei_proj/CT_test_ratescore.npy' \
--save_results_every 10 \
--save_model_every 5000 \
--num_train_steps 10000 \
--warmup_steps 10000 \
--lr 1e-4 \
--batch_size 5 \
--num_workers 8 \
--soft_label \
--use_triplet_loss