import random

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
    parser.add_argument("--pretrained_model_path", type=Path, help="Pretrained model path.", default=None)
        
    parser.add_argument(
        "--data_root",
        type=Path,
        default="./hw1_dataset/",
    )

    parser.add_argument(
        "--train_dir_prefix",
        type=Path,
        default="train/",
    )

    parser.add_argument(
        "--train_ann_file_name",
        type=Path,
        default="_annotations.coco.json",
    )

    parser.add_argument(
        "--val_dir_prefix",
        type=Path,
        default="valid/",
    )

    parser.add_argument(
        "--val_ann_file_name",
        type=Path,
        default="_annotations.coco.json",
    )

    parser.add_argument(
        "--test_dir_prefix",
        type=Path,
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
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")

    args = parser.parse_args()
    return args

def main(args):
    # Load config
    config = Config.fromfile("./configs/train_mmdet.py")
    config.model = args.model
    config.lr = args.lr
    config.beta1 = args.beta1
    config.train_dir = args.train_dir
    config.val_dir = args.val_dir
    config.test_dir = args.test_dir
    config.train_batch_size = args.train_batch_size
    config.val_batch_size = args.val_batch_size
    config.num_workers = args.num_workers
    config.num_epoch = args.num_epoch
    config.loss_fn = args.loss_fn
    config.num_pics_generated = args.num_pics_generated
    config.add_ms_loss = args.add_ms_loss
    config.noise = args.noise

    # Create runner
    runner = Runner(config, args.ckpt_dir)
    runner.run()



if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)