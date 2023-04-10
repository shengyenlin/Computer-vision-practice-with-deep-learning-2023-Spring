from __future__ import print_function
import os
import random
import time
import subprocess
import datetime
import gc

#%matplotlib inline
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from tqdm import tqdm

#torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from dataset import ObjectDetectionDataset
from preprocess import Transformation

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
    parser.add_argument(
        "--train_dir",
        type=Path,
        help="Directory to the training dataset.",
        default="./hw1_dataset/train",
    )
    parser.add_argument(
        "--val_dir",
        type=Path,
        help="Directory to the valid dataset.",
        default="./hw1_dataset/valid",
    )

    parser.add_argument(
        "--test_dir",
        type=Path,
        help="Directory to the test dataset.",
        default="./hw1_dataset/test",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # model
    parser.add_argument("--model", type=str, default='CNN', help="CNN, transformer")

    # optimizer
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # training
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--loss_fn", type=str, default='BCE')
    parser.add_argument("--num_pics_generated", type=int, default=1000)
    parser.add_argument("--add_ms_loss", type=int, default=0)
    parser.add_argument("--noise", type=str, default='StandardNorm')

    args = parser.parse_args()
    return args

def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    # print(args)
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    model_save_path = args.ckpt_dir / str(date) / str(run_id)
    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    # define train, valid, test dataset / data loader
    transform = Transformation()
    train_ds = ObjectDetectionDataset(
        args.train_dir, mode='train',
        transform=transform.transform['train']
        )

    val_ds = ObjectDetectionDataset(
        args.val_dir, mode='val',
        transform=transform.transform['val']
        )
    
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    # d_train = next(iter(train_dl))
    # d_val = next(iter(val_dl))

    # print(d_train)
    # print(d_val)

    # define model
    

    # define loss function

    # define optimizer

    hist = {
        'epoch': [],
        'iter': [],
        'map': []
    }

    best = {
        'epoch': 0,
        'iter': 0,
        'map': np.inf
    }

    # start training
    iters = 0
    for epoch in range(args.num_epoch):
        # For each batch in the dataloader
        #for i, data in enumerate(tqdm(train_dl, leave=False, colour='green')):
        for i, data in enumerate(train_dl):
            hist['epoch'].append(epoch)
            hist['iter'].append(i)
            train_batch(data, netD, netG, optimizerD, optimizerG, loss_fn, args, hist)
            
            # valid the map every epoch

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (hist['epoch'][-1], args.num_epoch, i, len(train_dl),
                        hist['D_losses'][-1], hist['G_losses'][-1], 
                        hist['D(x)'][-1], hist['D(G(z1))'][-1], hist['D(G(z2))'][-1]),
                        end = ' '
                        )

                eval(fixed_noise, netD, netG, epoch, i, img_save_path_ind, img_save_path_grid, hist)
                #store best epoch, show in the end at store as training result
                save_best(netD, netG, best, hist, model_save_path)
                gc.collect()
                torch.cuda.empty_cache()

                # Record every 50 iters
                # TODO: add inception score
                with open(Path(img_save_path) / 'best.txt', 'w') as f:
                    print(best, file=f)
                    print(args_dict, file=f)
                f.close()

                
            iters += 1
            gc.collect()
            torch.cuda.empty_cache()

    #TODO: cmd line: ctrl + L OR clear tqdm output
    print(f"Finish model training, best training result:")
    print(best)
    plot_eval_graph(hist, img_save_path)

    with open(Path(img_save_path) / 'best.txt', 'w') as f:
        print(best, file=f)
        print(args_dict, file=f)
    f.close()

    best_save_path = os.path.join(img_save_path, 'best')
    os.mkdir(best_save_path)
    generate_pics_with_best_model(
        os.path.join(model_save_path, 'best.ckpt'), 
        best_save_path, fixed_noise, args)    

if __name__ == '__main__':
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)