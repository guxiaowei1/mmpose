import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import PIL.Image
import os.path as osp

def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
        

json_path = '/home/guxiaowei/mmpose-master/data/coco/annotations/keypoints_val.json'
obj = read_jsonfile(json_path)  # 解析一个标注文件
img_files = obj['images']
for img_info in img_files:
    img_path = img_info['file_name']
    img_path = os.path.join('/home/guxiaowei/mmpose-master/', img_path)
    new_img_path = os.path.join('/home/guxiaowei/mmpose-master/data/val/', img_path.split('/')[-1])
    img_json_path = os.path.join('/home/guxiaowei/mmpose-master/', img_path[:-4] + '.json')
    new_json_path = os.path.join('/home/guxiaowei/mmpose-master/data/val/', img_path.split('/')[-1][:-4] + '.json')
    print(img_info['file_name'])
    shutil.copy(img_path, new_img_path)
    shutil.copy(img_json_path, new_json_path)