# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import sys

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--test_img_dir", type=str)
    parser.add_argument("--out_json_path", type=str)
    parser.add_argument("--weight_percentage", type=int, default=3)
    parser.add_argument("--model_path", type=str, default="./cache/best/model_final.pth")
    parser.add_argument("--save_back_bone", action="store_true")
    parser.add_argument("--model_prefix_dir", type=str)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)

    if args.config_file == "source_model_with_da.yaml":
        model_prefix_dir = "models/source_with_da"
    elif args.config_file == "source_only_model.yaml":
        model_prefix_dir = "models/source_without_da"
    
    if args.weight_percentage == -1: # self-defined model path
        model_path = args.model_path
    else:
        if args.weight_percentage == 0:
            model_name = "model_0_percentage.pth"
        elif args.weight_percentage == 1:
            model_name = "model_33_percentage.pth"
        elif args.weight_percentage == 2:
            model_name = "model_66_percentage.pth"
        elif args.weight_percentage == 3:
            model_name = "model_100_percentage.pth"
        elif args.weight_percentage == 4:
            model_name = "model_100_percentage.pth"
        model_path = os.path.join(args.model_prefix_dir, model_name)

    print("Load model from:", model_path)
    _ = checkpointer.load(model_path)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    print("test_img_dir: ", args.test_img_dir)
    data_loaders_val = make_data_loader(
        cfg, is_train=False, 
        is_distributed=distributed, test_img_dir=args.test_img_dir
        )
    
    
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        prediction = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_path=args.out_json_path,
            testing=True,
            save_back_bone=args.save_back_bone,
        )
        synchronize()


if __name__ == "__main__":
    main()
