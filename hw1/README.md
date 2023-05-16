# Computer vision practice in deep learning - HW1
- Graduate program of artificial intelligence, department of computer science and information engineering
- r11922a05 林聖硯

## Environment
- python version: 3.10.10
- pytorch version: 1.13.1 + cuda 11.7 + cudnn 8.3.2 (on my local server)

# MM Detection for inference
## Download checkpoint
```
bash hw1_download.sh
```

## How to install mmdetection
- Make sure that you have installed pytorch (with CUDA) on your local machine, if it's not the case, please run
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
- Note that it's impossible to directly install it with `pip install -r requirements.txt` since MMCV contains C++ and CUDA extensions, thus depending on PyTorch in a complex way.
- Please follow the instruction below to install mmdetection

```
# install dependencies of mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# install mmcv from source
git clone https://github.com/open-mmlab/mmdetection.git # do not clone it again if you have one in your directory
cd mmdetection
pip install -v -e .
cd ..
```

## Install other dependencies
```
pip3 install torchmetrics tqdm
```

## Testing
```bash
bash hw1_download.sh
bash hw1.sh [--test_img_dir] [--pred.json]
```
## Check test result on valid set
```bash
python3 check_your_prediction_valid.py [--pred_json_path] [--target_path]
```


## Reproduce Training
1. download dataset to `./hw1_dataset`
2. manually delete the following label in training and validation set
```
{
    "id": 0,
    "name": "creatures",
    "supercategory": "none"
}
```
3. run the following code
```
mkdir pretrained_models
```
4. download pretrained model from this [link](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth) and place it to `pretrained_models/`

---
# YOLOv5
## Environment
```
git clone https://github.com/ultralytics/yolov5 # do not clone it again if you have one in your directory
cd yolov5
pip install -r requirements.txt
cd ..
```

## Preprocess to data format for YOLOv5
```
mkdir hw1_dataset/labels
python3 coco_to_yolo_format.py
```

## Training
```
python3 train.py --img-size 640 --batch 8 --epochs 30 --data cvdpl_hw1.yaml --weights yolov5m.pt
```

