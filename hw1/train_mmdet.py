import os
import random
import time
import datetime

#%matplotlib inline
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmengine
from mmengine import Config
from mmengine.runner import set_random_seed, Runner

import torch

# Set random seed for reproducibility
SEED = 5566
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed:", SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, help="Config file path.")
    parser.add_argument("--dataset_type", type=str, help="Dataset type.", default='CocoDataset')
    parser.add_argument("--pretrained_model_path", type=str, help="Pretrained model path.", default=None)
        
    parser.add_argument(
        "--data_root",
        type=Path,
        default="./hw1_dataset/",
    )

    parser.add_argument(
        "--train_dir_prefix",
        type=str,
        default="train/",
    )

    parser.add_argument(
        "--train_ann_file_name",
        type=str,
        default="_annotations.coco.json",
    )

    parser.add_argument(
        "--val_dir_prefix",
        type=str,
        default="valid/",
    )

    parser.add_argument(
        "--val_ann_file_name",
        type=str,
        default="_annotations.coco.json",
    )

    parser.add_argument(
        "--test_dir_prefix",
        type=str,
        help="Directory to the test dataset.",
        default="test/",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=7)

    # training
    parser.add_argument("--cuda_number", type=int, default=0)
    parser.add_argument("--n_epoch", type=int, default=12)


    args = parser.parse_args()
    return args

def main(args):
    # print(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")

    # Load config
    torch.cuda.set_device(args.cuda_number)
    
    # Get the index of the current CUDA device
    current_cuda_device = torch.cuda.current_device()
    current_cuda_device
    print(f"Using cuda {current_cuda_device}.")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.pretrained_model_path

    # ckpt settings
    work_dir = args.ckpt_dir / str(date) / str(run_id)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    # dataset settings
    cfg.dataset_type = args.dataset_type
    classes = ('fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray')

    # data loader setting
    cfg.train_dataloader['dataset']['type'] = cfg.dataset_type
    cfg.train_dataloader['dataset']['data_root'] = args.data_root
    cfg.train_dataloader['dataset']['ann_file'] = os.path.join(args.train_dir_prefix, args.train_ann_file_name)
    cfg.train_dataloader['dataset']['data_prefix']['img'] = args.train_dir_prefix
    cfg.train_dataloader['dataset']['metainfo'] = dict(classes = classes)
    cfg.train_dataloader['batch_size'] = args.train_batch_size

    cfg.val_dataloader['dataset']['type'] = cfg.dataset_type
    cfg.val_dataloader['dataset']['data_root'] = args.data_root
    cfg.val_dataloader['dataset']['ann_file'] = os.path.join(args.val_dir_prefix, args.val_ann_file_name)
    cfg.val_dataloader['dataset']['data_prefix']['img'] = args.val_dir_prefix
    cfg.val_dataloader['dataset']['metainfo'] = dict(classes = classes)
    
    cfg.test_dataloader['dataset']['type'] = cfg.dataset_type
    cfg.test_dataloader['dataset']['data_root'] = args.data_root
    cfg.test_dataloader['dataset']['data_prefix']['img'] = args.test_dir_prefix
    cfg.test_dataloader['dataset']['metainfo'] = dict(classes = classes)

    # evaluator setting
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=os.path.join(args.data_root, args.val_dir_prefix, args.val_ann_file_name),
        metric=['bbox'],  # Metrics to be evaluated
        format_only=False,  # Only format and save the results to coco json file
        outfile_prefix= str(work_dir / 'valid'),  # The prefix of output file
    )
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=None,
        metric=['bbox'],  # Metrics to be evaluated
        format_only=True,  # Only format and save the results to coco json file
        outfile_prefix= str(work_dir / 'test'),  # The prefix of output file
    )

    # model
    cfg.model.backbone['init_cfg'] = None
    cfg.model.bbox_head.num_classes = args.num_classes

    # training settings
    cfg.train_cfg = dict(
        type='EpochBasedTrainLoop', 
        max_epochs=args.n_epoch, val_interval=1
        )

    # print("Config:")
    # print(cfg.pretty_text)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    print("Start training...")
    runner.train()

    print("Start testing...")
    runner.test()

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)