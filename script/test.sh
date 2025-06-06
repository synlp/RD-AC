
#acc: 74.6 AUROC: unknown
MODEL=../output_model/qwen-hmc-base
CHECK=../output_model/qwen-hmc-base/checkpoint-3645
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=1 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --data_name hmc \
    --output_file hmc-base.json

MODEL=../output_model/qwen-hmc-base-cls
CUDA_VISIBLE_DEVICES=1 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --model_path $MODEL \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --data_name hmc \
    --output_file hmc-base-cls.json

#acc: 74.0 AUROC: unknown
MODEL=../output_model/qwen-hmc-lmr-attention
CHECK=../output_model/qwen-hmc-lmr-attention/checkpoint-3645
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=2 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/hmc \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/hmc/all_data.json \
    --data_name hmc \
    --output_file hmc-lmr-attention.json

#### twitter
# acc: 77.91 F1: 71.42
MODEL=../output_model/qwen-twitter2015-base
CHECK=../output_model/qwen-twitter2015-base/checkpoint-1365
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=0 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/IJCAI2019_data/twitter2015_images \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/twitter2015/test_emb \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/twitter2015/all_data.json \
    --data_name twitter \
    --output_file twitter2015-base.json

# acc: 78.01 F1: 71.49
MODEL=../output_model/qwen-twitter2015-lmr-attention
CHECK=../output_model/qwen-twitter2015-lmr-attention/checkpoint-1365
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=0 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/IJCAI2019_data/twitter2015_images \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/twitter2015/test_emb \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/twitter2015/all_data.json \
    --data_name twitter \
    --output_file twitter2015-lmr-attention.json

#MSED
#
MODEL=../output_model/qwen-msed-base
CHECK=../output_model/qwen-msed-base/checkpoint-2628
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=3 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/MSED/ \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/MSED/test \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/MSED/all_data.json \
    --data_name msed \
    --output_file msed-base.json


MODEL=../output_model/qwen-msed-lmr-attention
CHECK=../output_model/qwen-msed-lmr-attention/checkpoint-2628
cp $MODEL/*.json $CHECK/
CUDA_VISIBLE_DEVICES=3 python test.py \
    --image_dir /media/ubuntu/data/yuanhe/project/affective_computing/data/MSED/ \
    --lmr_dir /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/MSED/test \
    --model_path $CHECK \
    --input_json /media/ubuntu/data/yuanhe/project/affective_computing/processed_data/MSED/all_data.json \
    --data_name msed \
    --output_file msed-base.json

