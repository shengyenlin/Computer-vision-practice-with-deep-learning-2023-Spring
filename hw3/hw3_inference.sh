python3 ./MIC/det/tools/test_net.py \
    --config-file source_model_with_da.yaml \
    --model_prefix_dir adapted_models \
    --test_img_dir $1 \
    --out_json_path $2 \
    --weight_percentage $3;