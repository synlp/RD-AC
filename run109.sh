
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 7 --master-port 21443 train.py \
    --model_name_or_path /media/ubuntu/data/share/Qwen2-VL-2B-Instruct \
    --training_data_path /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --training_image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --training_lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --data_name hmc \
    --output_dir ../output_model/qwen-hmc-lmr-attention-cls-2 \
    --save_total_limit 8 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 8 \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --bf16 true \
    --resume_from_checkpoint False \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 50 \
    --use_lmr \
    --use_attention

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 7 --master-port 21443 train.py \
    --model_name_or_path /media/ubuntu/data/share/Qwen2-VL-2B-Instruct \
    --training_data_path /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --training_image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --training_lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --data_name hmc \
    --output_dir ../output_model/qwen-hmc-base-cls-3 \
    --save_total_limit 5 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 5 \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --bf16 true \
    --resume_from_checkpoint False \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 50



CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path /media/ubuntu/data/share/Qwen2-VL-2B-Instruct \
    --training_data_path /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --training_image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --training_lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --data_name hmc \
    --output_dir ../output_model/qwen-hmc-base-cls \
    --save_total_limit 8 \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 8 \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --bf16 true \
    --resume_from_checkpoint False \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 50 \
    --small_data
