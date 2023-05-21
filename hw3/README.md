# Computer vision practice in deep learning - HW3
- Graduate program of artificial intelligence, department of computer science and information engineering
- r11922a05 林聖硯

## Environment
- python version: 3.10.10
- pytorch version: 1.13.1 + cuda 11.7 + cudnn 8.3.2 (on my local server)

# MIC for domain adaptation

## Environment setting
```bash
conda create -n cvdpl-hw3-mic python=3.10 -y

# pytorch (please modify the version of pytorch and dependencies according to your local machine)
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
pip3 install -r requirements.txt

unset INSTALL_DIR
```

## Reproduce training

Please refer to `train.md`

**Warning: The training process may take over 20 hours**

## Reproduce inference

- Download checkpoint
```
bash hw3_download.sh
```
After running the code above, please check if there's a directory called `adapted_models` and there should be 4 model checkpoints inside this directory - `model_O_percentage.pth`, `model_33_percentage.pth`, `model_66_percentage.pth`, `model_100_percentage.pth`

- Inference on source valid/test sets
```bash
bash hw3_inference_source_model.sh [--test_source_img_dir] [--output_json_path] [--checkpoint_number]
```

- Inference on target valid/test sets w/o DA
```bash
bash hw3_inference_source_model.sh [--test_target_img_dir] [--output_json_path] [--checkpoint_number]
```

- Inference on target valid/test sets w/ DA
```bash
bash hw3_inference.sh [--test_target_img_dir] [--output_json_path] [--checkpoint_number]
```

### Flags
- test_source_img_dir: directory to testing source images
- test_target_img_dir: directory to testing target images
- output_json_path: path to predicted object detection json file
- checkpoint_number: one of the numbers between 0 and 3, specifying which checkpoint to use, e.g., 0 indicates the 0% checkpoint, and 3 indicates the 100% checkpoint.
  - If you want to use the best model to inference, please use 4 as the check point number. In my case, the best model is the 100% checkpoint model.
  - If you want to use weights other than the default 4 checkpoint_number, set the checkpoint_number to -1 and use `--model_path` flag to specify your model path, and directory run the py file in command line.

E.g.
```bash
python3 ./MIC/det/tools/test_net.py \
    --config-file source_only_model.yaml 
    --test_img_dir [--test_img_dir] \
    --out_json_path [--out_json_path] \
    --weight_percentage -1 \
    --model_path [--your_own_model_path];
```

## Check predicted json result
```bash
python3 check_your_prediction_valid.py [--pred_json_path] [--gt_json_path]
```
