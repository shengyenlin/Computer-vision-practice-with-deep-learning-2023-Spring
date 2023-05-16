import random
import sys
import os
import json
import time

import torch

from mmengine import Config
from mmdet.apis import init_detector, inference_detector

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

test_img_dir = sys.argv[1]
out_json_file = sys.argv[2]

config_file = 'dino_config.py'
checkpoint_file = 'dino_model.pth'

def post_process_label(label):
    return label + 1

use_threshold = False
threshold_value = 0.3

def main():
    x = time.time()
    # load model
    model = init_detector(
        config_file,
        checkpoint_file,
        device='cuda:0')

    # get filenames in test_img_dir
    imgs = os.listdir(test_img_dir)

    out_json = dict()
    for img_name in imgs:
        img_path = os.path.join(test_img_dir, img_name)
        result = inference_detector(model, img_path)

        # bboxes: (xmin, ymin, xmax, ymax)
        scores, bboxes, labels = \
            result.pred_instances.scores.tolist(), \
                result.pred_instances.bboxes.tolist(), \
                    result.pred_instances.labels.tolist()
        
        labels = [post_process_label(label) for label in labels]

        # Threshold
        if use_threshold:
            indices_above_threshold = [i for i, x in enumerate(scores) if x > threshold_value]
            scores = [scores[i] for i in indices_above_threshold]
            bboxes = [bboxes[i] for i in indices_above_threshold] 
            labels = [labels[i] for i in indices_above_threshold]

        ans = {"boxes": bboxes, "labels": labels, "scores": scores}
        out_json[img_name] = ans
    
    # save to json file
    with open(out_json_file, 'w') as f:
        json.dump(out_json, f, indent=4)
    
    y = time.time()
    print("Finish testing in {} minutes".format((y-x)/60))

if __name__ == "__main__":
    main()