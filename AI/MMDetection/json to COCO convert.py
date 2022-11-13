import json
import collections
import glob
import os
import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO

#폴더내 json파일을 하나의 파일로 합치는 부분.
path_dir = 'data/train_road_information/label'
file_list = os.listdir(path_dir)
merged = ""
p = open('data/train_road_information/label/merged.json', 'w')
for i in file_list :
    with open(path_dir+'/'+i, 'r') as f:
        data = f.read()
        merged += data + '\n'

p.write(merged)



#하나로 합친json파일을 coco파일로 변환하는 부분.
coco_group = OrderedDict()
categories = OrderedDict()
images = OrderedDict()
annotations = OrderedDict()


images = {}
annotations = {}
#category 설정
coco_group["categories"] = []
coco_group["images"] = []
coco_group["annotations"] = []

coco_group["categories"].append({
    "id": 0,
    "name": "traffic_light"
})
coco_group["categories"].append({
    "id": 1,
    "name": "traffic_sign"
})
coco_group["categories"].append({
    "id": 2,
    "name": "traffic_information"
})

print(json.dumps(coco_group, ensure_ascii=False, indent="\t"))

image_name = 'data/train_road_information/image'
find_ext = '*jpg'

images_list = []
ann_list = []
category_id = []
category_id_set = []
inner1_list = []

inner1 = {}
with open('data/train_road_information/label/merged.json', 'r+') as f:
    json_list = f.readlines()
    print(len(json_list))
#annotation 데이터 처리부분
annotation_id = 0
img_id = 0
for i in np.arange(len(json_list)):
    if( i  >= len(json_list)-1):
        break

    data = json.loads(json_list[i])

    # annotation Data 추출 & 가공파트
    annotation_Data = data['annotation']
    for j in np.arange(len(annotation_Data)):
        category_id = 0 # 기본은 0 traffic light
        if(annotation_Data[j]['class'] == "traffic_sign"):
            category_id = 1 # sign이면 1로 변환
        coco_group["annotations"].append({"segmentation": [[0]],
                                          "image_id": img_id,
                                          "bbox": [annotation_Data[j]["box"][0],annotation_Data[j]["box"][1],annotation_Data[j]["box"][2],annotation_Data[j]["box"][3]],
                                          "category_id": category_id,
                                          "id": annotation_id,
                                          "area": annotation_Data[j]["box"][2] * annotation_Data[j]["box"][3],#Width * Height
                                          "iscrowd": 0}) # 0으로 고정
        annotation_id = annotation_id+1
    # image_id추출 & 가공파트
    img_Data = data['image']
    coco_group["images"].append(({
        "file_name": img_Data['filename'],
        "width": img_Data['imsize'][0],
        "height": img_Data['imsize'][1],
        "id": img_id
    }))
    img_id = img_id + 1




print(json.dumps(coco_group, ensure_ascii=False, indent="\t"))
with open('data/coco_road_info.json', 'w') as outfile:
    json.dump(coco_group, outfile, indent="\t")
print(len(coco_group['categories']))









