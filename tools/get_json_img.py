import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import PIL.Image
import os.path as osp
from tqdm import tqdm
import json
from base64 import b64encode
from json import dumps
import shutil

def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
        
if __name__ == '__main__':    
    jpg_dict = {}
    src_obj = read_jsonfile('/home/guxiaowei/mmpose-master/data/coco/annotations/keypoints_val.json')
    for i in range(len(src_obj['images'])):
        src_img_path = src_obj['images'][i]['file_name']
        dst_img_path = src_img_path.replace('modify_data', 'img')
        shutil.copy(src_img_path, dst_img_path)
    