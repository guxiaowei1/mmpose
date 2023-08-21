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
    json_match_dict = {}
    src_json_file = '/home/guxiaowei/mmpose-master/data/modify_data/'
    for root, dir_list, file_list in os.walk(src_json_file):
            for index, file_fn in enumerate(file_list):
                if file_fn.endswith('json'):
                    json_match_dict[file_fn] = []
    
    cut_json_file = '/home/guxiaowei/mmpose-master/data/cut_img/'          
    for root, dir_list, file_list in os.walk(cut_json_file):
            for index, file_fn in enumerate(file_list):
                if file_fn.endswith('json'):
                    if file_fn.split('_JYZ')[0] + '.json' in json_match_dict:
                        json_match_dict[file_fn.split('_JYZ')[0] + '.json'].append(file_fn)

    for src_json_path, cut_json_path in json_match_dict.items():
            print(src_json_path)
            print(cut_json_path)
            # if not cut_json_path or os.path.exists(os.path.join(src_json_file, src_json_path).replace('images','modify_data')):
            if not cut_json_path:
                continue
            bboxes_list, keypoints_list = [], []
            src_obj = read_jsonfile(os.path.join(src_json_file, src_json_path))  # 解析一个标注文件
            for i in range(len(cut_json_path)):
                cut_obj = read_jsonfile(os.path.join(cut_json_file, cut_json_path[i]))
                cut_shapes = cut_obj['shapes']
                for cut_shape in cut_shapes:
                    if cut_shape['shape_type'] == 'point':
                        keypoints_list.append(cut_shape)
                
            shapes = src_obj['shapes']  # 读取 labelme shape 标注
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':  # bboxs
                    bboxes_list.append(shape)           # keypoints
                    
            json_dict = {
                    "version": "4.5.7",
                    "flags": {},
                    "shapes": [],
                    "imageHeight": src_obj['imageHeight'],
                    "imageWidth": src_obj['imageWidth']
                }
            for rect_ind in range(len(bboxes_list)):
                bbox_dict = bboxes_list[rect_ind]
                bbox_label = bbox_dict['label']
                if bbox_label != 'JYZ':
                    bbox_label = 'JYZ'
                bbox_points = bbox_dict['points']  
                bbox_group_id = bbox_dict['group_id']
                bbox_shape_type = bbox_dict['shape_type']            
                json_dict['shapes'].append({
                    "label": bbox_label,
                    "points": bbox_points,
                    "group_id": bbox_group_id,
                    "shape_type": bbox_shape_type,
                    "flags": {}
                    })
            for point_ind in range(0, len(keypoints_list), 2):
                p2b_ind = point_ind // 2
                p2bbox_dict = bboxes_list[p2b_ind]
                rect_cord = p2bbox_dict['points']
                x_left, y_left = min(rect_cord[0][0], rect_cord[1][0]), min(rect_cord[0][1], rect_cord[1][1])
                keypoint1_dict = keypoints_list[point_ind]
                keypoint1_label = keypoint1_dict['label']
                keypoint1_group_id = keypoint1_dict['group_id']
                keypoint1 = keypoint1_dict['points'][0]
                json_dict['shapes'].append({
                    "label": keypoint1_label,
                    "points": [
                        [
                        keypoint1[0] + x_left,
                        keypoint1[1] + y_left
                        ]
                    ],
                    "group_id": keypoint1_group_id,
                    "shape_type": "point",
                    "flags": {}
                })
                
                keypoint2_dict = keypoints_list[point_ind + 1]
                keypoint2_label = keypoint2_dict['label']
                keypoint2_group_id = keypoint2_dict['group_id']
                keypoint2 = keypoint2_dict['points'][0]
                json_dict['shapes'].append({
                    "label": keypoint2_label,
                    "points": [
                        [
                        keypoint2[0] + x_left,
                        keypoint2[1] + y_left
                        ]
                    ],
                    "group_id": keypoint2_group_id,
                    "shape_type": "point",
                    "flags": {}
                })
            
            img_path = os.path.join(src_json_file, src_json_path)
            if os.path.exists(img_path.replace('.json','.jpg')):
                img_path = img_path.replace('.json','.jpg')
            elif os.path.exists(img_path.replace('.json','.JPG')):
                img_path = img_path.replace('.json','.JPG')
            else:
                img_path = img_path.replace('.json','.png')
               
            with open(img_path, 'rb') as jpg_file:
                byte_content = jpg_file.read()
        
                # 把原始字节码编码成base64字节码
                base64_bytes = b64encode(byte_content)
            
                # 把base64字节码解码成utf-8格式的字符串
                base64_string = base64_bytes.decode('utf-8')
        
            # 用字典的形式保存数据
            json_dict["imageData"] = base64_string
            json_dict["imagePath"] = img_path

            # shutil.copy(img_path, img_path.replace('images', 'modify_data'))
            with open(os.path.join('/home/guxiaowei/mmpose-master/data/modify_data/', src_json_path), "w", encoding='utf-8') as f:
                # json.dump(dict_var, f)  # 写为一行
                json.dump(json_dict, f,indent=2,sort_keys=False, ensure_ascii=False)  # 写为多行
            
