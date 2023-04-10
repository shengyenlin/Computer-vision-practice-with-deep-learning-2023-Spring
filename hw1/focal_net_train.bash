python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  tools/train.py \
configs/focalnet/mask_rcnn_focalnet_base_patch4_mstrain_480-800_adamw_1x_coco.py \
--cfg-options \
model.pretrained='focalnet_base_lrf.pth' \
data.samples_per_gpu=2 \
model.backbone.focal_levels='[3,3,3,3]' \
--launcher pytorch