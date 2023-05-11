# Computer vision practice in deep learning - HW3
- Graduate program of artificial intelligence, department of computer science and information engineering
- r11922a05 林聖硯

## Environment
- python version: 3.10.10
- pytorch version: 1.13.1 + cuda 11.7 + cudnn 8.3.2 (on my local server)

# MIC for domain adaptation

- TODO: add some description for MIC

## Download checkpoint
```
bash hw3_download.sh
```

## Environment setting
```bash
conda create -n cvdpl-hw3-mic python=3.10 -y

# pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# maskrcnn_benchmark and coco api dependencies
pip install ninja==1.10.2.3 yacs==0.1.8 cython==0.29.28 matplotlib==3.5.1 tqdm==4.63.0 opencv-python==4.5.5.64

# MIC dependencies
pip install timm==0.6.11 kornia==0.5.8 einops==0.4.1

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install MIC - object detection
cd $INSTALL_DIR
git clone https://github.com/lhoyer/MIC
cd MIC/det
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it

python setup.py build develop

pip3 install numpy==1.23

unset INSTALL_DIR
```

## Reproduce training
- train source model
```bash
python3 ./MIC/det/tools/train_net.py --mode source_only
```
- train source model with MIC
```bash 
python3 ./MIC/det/tools/train_net.py --mode source_with_da
```

TODO: 
- modify the output path of training model
- add what the user will see after reproducing training result

## Reproduce inference
- source valid/test sets
```bash
bash hw3_inference_source_model.bash [--test_source_img_dir] [--output_json_path] [--checkpoint_number]
```

- target valid/test sets w/o DA
```bash
bash hw3_inference_source_model.bash [--test_target_img_dir] [--output_json_path] [--checkpoint_number]
```

- target valid/test sets w/ DA
```bash
bash hw3_inference.bash [--test_target_img_dir] [--output_json_path] [--checkpoint_number]
```

### Flags
- test_source_img_dir: directory to testing source images
- test_target_img_dir: directory to target source images
- output_json_path: path to predicted object detection json file
- checkpoint_number: one of the numbers between 0 and 3, specifying which checkpoint to use, e.g., 0 indicates the 0% checkpoint, and 3 indicates the 100% checkpoint.

## Check predicted json result
```bash
python3 check_your_prediction_valid.py [--pred_json_path] [--target_path]
```
