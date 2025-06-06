
python compute_matrix_hmc.py \
    --data_dir /path/to/data/hmc \
    --output_dir /path/to/processed_data/hmc \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --clip_model_path /path/to/clip-vit-base-patch32


python main_hmc.py \
    --input_dir /path/to/processed_data/hmc \
    --num_workers 40  \
    --lambda_param 1 \
    --mu 10 \
    --max_iter 3000 \
    --tol 1e-6 \
    --min_iter 5 \
    --output_json all_data.json

#python translate_hmc.py
