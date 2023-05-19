# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os
import json

import numpy as np
import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize

def compute_on_dataset(dataset, model, data_loader, device, save_back_bone=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    backbone_features_list = []
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output, backbone_features = model(images)
            output = [o.to(cpu_device) for o in output]
            if save_back_bone:
                backbone_features_list.append(
                    # choose the lowest level feature map
                    # [1, 256, 13, 25] -> [1, 256*13*25]
                    backbone_features[4].cpu().numpy().flatten()
                )
                
        # for img_id, result in zip(image_ids, output):
        #     img_info = dataset.get_img_info(img_id)
        #     result.add_field("img_info", img_info)
        #     results_dict.update(
        #         {img_id: result}
        #     )
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

    if save_back_bone:
        np.savetxt(
            'backbone_features_source_clear.tsv', 
            backbone_features_list,
            delimiter='\t'
            )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_path=None,
        testing=False,
        save_back_bone=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    
    #### The prediction output #####
    # prediction: dict_keys(['boxes', 'labels', 'scores', 'masks', 'keypoints', 'img_info'])
    predictions = compute_on_dataset(dataset, model, data_loader, device, save_back_bone=save_back_bone)
    #### The prediction output #####

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_path:
        out_json = dict()

        for image_id, prediction in enumerate(predictions):
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            image_name = img_info["file_name"]
            prediction = prediction.resize((image_width, image_height))

            bboxes = prediction.bbox.tolist()  # get the bounding boxes as a list of lists
            labels = prediction.get_field('labels').tolist()  # get the labels as a list
            scores = prediction.get_field('scores').tolist()  # get the scores as a list

            ans = {"boxes": bboxes, "labels": labels, "scores": scores}
            out_json[image_name] = ans

        with open(output_path, 'w') as f:
            json.dump(out_json, f)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    if testing is True:
        return predictions
    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_path,
                    **extra_args)
