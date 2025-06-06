# RD-AC

This is the code for our paper titled "Representation Decomposition for Learning Similarity and Contrastness Across Modalities for Affective Computing".

## Train

### Get the low-rank matrix

See the commands in `command.sh` for examples to run on HMC data.

### Train the model

Run the following command to train a model 

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes 1 --nproc_per_node 8 --master-port 21443 train.py \
    --model_name_or_path /path/to/Qwen2-VL-2B-Instruct \
    --training_data_path /path/to/processed_data/hmc/all_data.json \
    --training_image_dir /path/to/data/hmc \
    --training_lmr_dir path/to/processed_data/hmc \
    --data_name hmc \
    --output_dir /path/to/output_model \
    --save_total_limit 1 \
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
```

### Test the model

Run the following command to test a model

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --image_dir /path/to/data/hmc \
    --lmr_dir /path/to/processed_data/hmc \
    --model_path /path/to/output_model \
    --input_json /path/to/processed_data/hmc/all_data.json \
    --data_name hmc \
    --output_file hmc-base.json
```

