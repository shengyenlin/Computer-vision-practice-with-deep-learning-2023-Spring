python3 train_mmdet.py \
    --config ./mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
    --pretrained_model_path ./pretrained_models/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth \
    --n_epoch 30 \
    --train_batch_size 2 \
    --cuda_number 0;