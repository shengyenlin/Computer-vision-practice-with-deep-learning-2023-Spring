import json
import sys

file_name = sys.argv[1]
out_file_name = sys.argv[2]

def main():
    with open(file_name, 'r') as f:
        file = json.load(f)
    img_info = file['images']
    num_imgs = len(file['images'])

    combine_data = []
    # start to loop 
    cur_image_id = 0
    num_obj_cnt = 0
    bboxes, category_ids, areas, iscrowds = [], [], [], []
    for data in file['annotations']:
            img_data = {}
            if cur_image_id == data['image_id']:
                num_obj_cnt += 1
                bboxes.append(data['bbox'])
                category_ids.append(data['category_id'])
                areas.append(data['area'])
                iscrowds.append(data['iscrowd'])

            else:
                img_data["id"] = cur_image_id
                img_data["num_obj"] = num_obj_cnt
                img_data["bbox"] = bboxes
                img_data["category_id"] = category_ids
                img_data["area"] = areas
                img_data["iscrowd"] = iscrowds
                combine_data.append(img_data)

                cur_image_id += 1
                num_obj_cnt = 0
                bboxes, category_ids, areas, iscrowds = [], [], [], []

    # last image
    img_data["id"] = cur_image_id
    img_data["num_obj"] = num_obj_cnt
    img_data["bbox"] = bboxes
    img_data["category_id"] = category_ids
    img_data["area"] = areas
    img_data["iscrowd"] = iscrowds
    combine_data.append(img_data)

    new_data = []
    for i in range(num_imgs):
         new_data.append({**img_info[i], **combine_data[i]})

    # write to json file
    with open(out_file_name, 'w') as f:
        json.dump(new_data, f)

if __name__ == '__main__':
    main()