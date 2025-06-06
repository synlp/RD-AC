
python compute_matrix_mabsa.py \
    --data_dir /path/to/data/IJCAI2019_data/twitter2015 \
    --img_home_dir /path/to/data/IJCAI2019_data/twitter2015_images \
    --output_dir /path/to/processed_data/twitter2015 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --clip_model_path /path/to/clip-vit-base-patch32

python main_absa.py \
    --data_dir /path/to/processed_data/twitter2015 \
    --num_workers 40 \
    --lambda_param 1 \
    --mu 10 \
    --max_iter 3000 \
    --tol 1e-6 \
    --min_iter 5 \
    --output_json all_data.json

