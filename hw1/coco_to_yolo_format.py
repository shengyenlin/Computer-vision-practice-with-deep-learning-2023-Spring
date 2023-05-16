import json
import os
from collections import defaultdict
from pathlib import Path

def generate_yolo_labels(dataset_path: Path, folder: str) -> None:
    os.makedirs(dataset_path / "labels" / folder, exist_ok=True)

    ann_path = dataset_path / "images" / folder / "_annotations.coco.json"
    with ann_path.open() as f:
        data = json.load(f)

    categories = data['categories']
    images = data['images']

    # Create a dict that maps image_id to file name 
    image_id_to_name = {image['id']: image['file_name'] for image in images}

    # Group the annotations by their image_id
    # i.e. convert to a dictionary, where (key, value) = (image_id, list of annotations)
    annotations_dict = defaultdict(list)
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        annotations_dict[image_id].append(annotation)

    # Delete those images that have no annotations
    images = [img for img in images if img['id'] in annotations_dict]

    # For each image, create a *.txt file
    label_folder_path = dataset_path / "labels" / folder
    print(f"Generating labels for {folder} data. Save to {label_folder_path}")
    for image in images:
        image_id = image['id']
        image_name = image_id_to_name[image_id]
        if not image_name.endswith(".jpg"):
            raise ValueError(f"Image name {image_name} does not end with .jpg")
        image_name = image_name[:-4] + ".txt"
        image_annotations = annotations_dict[image_id]
        image_path = label_folder_path / image_name
        with image_path.open(mode="w") as f:
            for annotation in image_annotations:
                box = annotation['bbox']
                class_id = annotation['category_id']
                # Convert box coordinates (xmin, ymin, w, h) to normalized xywh format
                x_center = (box[0] + box[2] / 2) / image['width']
                y_center = (box[1] + box[3] / 2) / image['height']
                width = box[2] / image['width']
                height = box[3] / image['height']
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def main() -> None:
    folders = ["train", "valid"]
    dataset_path = Path("hw1_dataset")

    for folder in folders:
        generate_yolo_labels(dataset_path, folder)


if __name__ == "__main__":
    main()
