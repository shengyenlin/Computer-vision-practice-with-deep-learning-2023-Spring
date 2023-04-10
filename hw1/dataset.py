import os
import json
import copy
import glob

import numpy as np
import mmcv
from mmdet.registry import DATASETS
from mmdet.datasets.custom import CustomDataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class ObjectDetectionDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode # train, valid, test
        
        if self.mode == 'train':
            data_name = os.path.join(self.root_dir, 'train_annotations_organized.json')

        elif self.mode == 'val':
            data_name = os.path.join(self.root_dir, 'valid_annotations_organized.json')

        self.data = json.load(open(data_name, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        img_name = os.path.join(
            self.root_dir, 
            d["file_name"]
            )
        img = Image.open(img_name).convert("RGB")

        num_objs = d["num_obj"]
        boxes = d["bbox"]

        tgt = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(d["category_id"], dtype=torch.int64)

        print(boxes.shape, labels.shape)
        print(boxes)
        print(labels)
        tgt["boxes"] = boxes
        tgt["labels"] = labels
        
        if self.transform is not None:
            img, tgt = self.transform(img, tgt)

        return img, tgt
    

