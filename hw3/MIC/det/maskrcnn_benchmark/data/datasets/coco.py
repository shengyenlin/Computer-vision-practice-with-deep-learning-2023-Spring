# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import os

from PIL import Image
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    if len([obj for obj in anno if obj["iscrowd"] == 0])==0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None, is_source=True
    ):

        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids) # list of image ids

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        # self.coco.getCatIds() - list of category ids
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        } # category id to contiguous id
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.is_source = is_source

    def __getitem__(self, idx):
        # print(self.get_img_info)
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno] if self.is_source else [[0, 0, img.size[0]-1, img.size[1]-1]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno] if self.is_source else [self.contiguous_category_id_to_json_id[1]]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # TODO: check if it's has no effect on the performance
        # masks = [obj["segmentation"] for obj in anno] if self.is_source else [[[0, 0, 0, img.size[1]-1, img.size[0]-1, img.size[1]-1, img.size[0]-1, 0]]]
        # masks = SegmentationMask(masks, img.size)
        # target.add_field("masks", masks)

        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)
        target.add_field("is_source", domain_labels)

        if anno and "keypoints" in anno[0]:
            if self.is_source:
                keypoints = [obj["keypoints"] for obj in anno]
            else:
                raise NotImplementedError
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

class COCOTestingDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, transforms=None, is_source=True):
        super(COCOTestingDataset, self).__init__(root, transforms=transforms)
        self.root = root
        self.img_paths = glob.glob(os.path.join(root, "**/*.png"), recursive=True)
        # create a list of image ids, where each element correspoinds to the image path
        self.ids = [i for i in range(len(self.img_paths))] # (id - image path)

        # 1: person, 2: car, 3: truck, 4: bus, 5: rider,6: motorcycle, 7: bicycle, 8: train
        self.cats_ids = [i+1 for i in range(8)]
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.cats_ids)
        } 

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.is_source = is_source

    def get_img_info(self, idx):

        img_path = self.img_paths[idx]
        img_id = self.ids[idx]
        img = Image.open(
            # os.path.join(self.root, img_path)
            img_path
            ).convert("RGB")

        # "id", "width", "height", "file_name"
        img_data = {
            "id": img_id,
            "width": img.size[0],
            "height": img.size[1],
            "file_name": os.path.relpath(img_path, self.root)
        }
        return img_data

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(
            # os.path.join(self.root, img_path)
            img_path
            ).convert("RGB")
        
        img_path_without_root = os.path.relpath(img_path, self.root)

        ########### Random things for testing ###########
        # add random bbox
        boxes = [[0, 0, img.size[0]-1, img.size[1]-1]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # add random classes
        classes = [self.contiguous_category_id_to_json_id[1]]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # add random domain labels
        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)
        target.add_field("is_source", domain_labels)
        ##################################################

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        # return img_path_without_root, img, target, idx
        return img, target, idx

    def __len__(self):
        return len(self.img_paths)