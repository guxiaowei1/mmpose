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
import xml.etree.ElementTree as ET
import cv2
from xml.etree.ElementTree import Element
from natsort import natsorted
import re


def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

def xml2json(src_xml_file, cut_json_file, json_file): 
    json_match_dict, json_sort = {}, {}
    for root, dir_list, file_list in os.walk(src_xml_file):
            for index, file_fn in enumerate(file_list):
                img_file1, img_file2 = os.path.join(src_xml_file, file_fn.replace('xml', 'JPG')),os.path.join(src_xml_file, file_fn.replace('xml', 'jpg'))
                if file_fn.endswith('xml') and (os.path.exists(img_file1) or (os.path.exists(img_file2))):
                    json_match_dict[file_fn] = []
                    json_sort[file_fn] = []
      
    for root, dir_list, file_list in os.walk(cut_json_file):
            for index, file_fn in enumerate(file_list):
                if file_fn.endswith('json'):
                    # if file_fn.split('_JYZ')[0] + '.xml' in json_match_dict:
                    if file_fn.split('_JYZ')[0] + '.xml' in json_match_dict:
                        cut_obj = read_jsonfile(os.path.join(cut_json_file, file_fn))
                        if cut_obj['shapes'] != []:
                            json_match_dict[file_fn.split('_JYZ')[0] + '.xml'].append(file_fn)
                            json_sort[file_fn.split('_JYZ')[0] + '.xml'].append(int(re.findall('\d+', file_fn.split('_JYZ')[1])[0]))
                        else:
                            print(cut_obj)

    for src_xml_path, cut_json_path in json_match_dict.items():
            cut_json_path = natsorted(cut_json_path, key=lambda x:re.findall('\d+', x.split('_JYZ')[1])[0])
            sort_num = []
            for cut_json_path_per in cut_json_path:
                sort_num.append(int(re.findall('\d+', cut_json_path_per.split('_JYZ')[1])[0]))
            if sort_num == sorted(sort_num):
                pass
            else:
                raise Exception("found error object name! Name is:", object_name, "in file:", cut_json_path)
            print(src_xml_path)
            print(cut_json_path)
            if not cut_json_path:
                continue
            bboxes_list, keypoints_list = [], []

            shapes = []
            bndbox = dict()
            size = dict()
            file_name = None
            count = 0
            size['width'] = None
            size['height'] = None
            size['depth'] = None
            tree = ET.parse(os.path.join(src_xml_file, src_xml_path))
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('root not annotation')

            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None
                shape = {}

                if elem.tag == 'folder' or elem.tag == 'source' or elem.tag == 'segmented':
                    continue

                if elem.tag == 'filename':
                    file_name = elem.text
                
                if elem.tag == 'path':
                    img_path = elem.text.split("\\")[-1]
                    path = os.path.join(src_xml_file, img_path)
                    ori_img = cv2.imread(path)

                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['ymin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name in ["JYZ", "JYZK", "jyz", 'JYZ_K', "JYZ_XS", "JYZ_ZC"]:
                            # object_name = "JYZ"
                            count += 1 
                        elif object_name in ["CTJYZ", "TM", "GLKG", "DLSRDQ", "BYQ", "DLSBLQ","BLZRDQ", 'GTPS', 'naizhangxianjia', 'bangzadai', 'PS', 'ps',\
                            'bangzadai_QS', 'jueyuanzi_PS','bangzidai','juyuanzi_PS','gtps','GTQS', 'juyuanzi','bangzadaiQS','bangzaidai_QS','`', 'jueyuanzi_QS','GDPS'\
                                ,'yueyuanzi','bangzadai_PS','jueyuanzips','bangzadai_qs','jueyuanzi_ps', 'JYZK', 'JYZ','jyz', 'BYQ_K' 'GANTOU_ZC',\
                                    'GANTOU_PS','TATOU','tatou','GANTA_QX','GANTA_ZC','GANTOU_ZC','BYQ_JYZ','JYZk','GANTA_PS','xdqxz','1','BHQ', 'JYZ_K'
                            ,'ZSKG','ZSGK', 'ZSKG_difficult','DLSRDQ','DLSRDQ_difficult','DLSBLQ','GLKG','GLKG_difficult','BYQ','BYQ_K','BYQ_difficult',\
                                'DLSBLQ_difficult','DLSRDQ_diffcult','tatou']:
                            break
                        else:
                            raise Exception("found error object name! Name is:", object_name, "in file:", os.path.join(src_xml_file, src_xml_path))
                        
                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structre broken at size tag')
                        size[subelem.tag] = int(subelem.text)

                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structre broken at bndbox tag')
                            bndbox[option.tag] = int(option.text)

                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structre broken at bndbox tag')
                        
                        size['width'] = bndbox['xmax'] - bndbox['xmin']
                        size['height'] = bndbox['ymax'] - bndbox['ymin']
                        new_path = cut_json_file  + img_path[:-4] + '_' + object_name +str(count) + '.json'
                        if os.path.exists(new_path):
                            if new_path.split('/')[-1] in cut_json_path:
                                shape['label'] = 'JYZ'
                                print(new_path)
                                if size['width'] > 1 and size['height'] > 1:
                                    x1, y1 = int(bndbox['xmin']), int(bndbox['ymin'])
                                    x2, y2 = int(bndbox['xmax']), int(bndbox['ymax'])
                                    w, h = x2 - x1, y2 - y1
                                    x1, y1 = max(0, int(x1 - w / 6)), max(0, int(y1 - h / 6))
                                    x2, y2 = min(ori_img.shape[1], int(x2 + w / 6)), min(ori_img.shape[0], int(y2 + h / 6))
                                    # cv2.imwrite(new_path, ori_img[y1:y2, x1:x2, :])
                                    shape['points'] = [[x1, y1], [x2, y2]]
                                    shape['group_id'] = None
                                    shape['shape_type'] = 'rectangle'
                                    shape['flags'] = {}
                                    shapes.append(shape)
                                else:
                                    print(new_path)
                            else:
                                print(new_path)
                      
                        bndbox['xmax'] -= bndbox['xmin']
                        bndbox['ymax'] -= bndbox['ymin']
                        bndbox['xmin'] = 0
                        bndbox['ymin'] = 0
            
            
            for i in range(len(cut_json_path)):
                cut_obj = read_jsonfile(os.path.join(cut_json_file, cut_json_path[i]))
                cut_shapes = cut_obj['shapes']
                for cut_shape in cut_shapes:
                    if cut_shape['shape_type'] == 'point':
                        keypoints_list.append(cut_shape)
                
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':  # bboxs
                    bboxes_list.append(shape)           # keypoints
                    
            json_dict = {
                    "version": "4.5.7",
                    "flags": {},
                    "shapes": [],
                    "imageHeight": ori_img.shape[0],
                    "imageWidth": ori_img.shape[1]
                }
            if len(bboxes_list) != len(keypoints_list) / 2:
                print(new_path)
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
            
            img_path = os.path.join(src_xml_file, src_xml_path)
            if os.path.exists(img_path.replace('.xml','.jpg')):
                img_path = img_path.replace('.xml','.jpg')
            elif os.path.exists(img_path.replace('.xml','.JPG')):
                img_path = img_path.replace('.xml','.JPG')
            else:
                img_path = img_path.replace('.xml','.png')
               
            with open(img_path, 'rb') as jpg_file:
                byte_content = jpg_file.read()
        
                # 把原始字节码编码成base64字节码
                base64_bytes = b64encode(byte_content)
            
                # 把base64字节码解码成utf-8格式的字符串
                base64_string = base64_bytes.decode('utf-8')
        
            # 用字典的形式保存数据
            json_dict["imageData"] = base64_string
            json_dict["imagePath"] = img_path

            shutil.copy(img_path, img_path.replace('img', 'xml2json'))
            # with open(os.path.join(json_file, src_xml_path.replace('xml', 'json')), "w", encoding='utf-8') as f:
            #     # json.dump(dict_var, f)  # 写为一行
            #     json.dump(json_dict, f,indent=2,sort_keys=False, ensure_ascii=False)  # 写为多行
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_xml_file', type=str, default="/home/guxiaowei/mmpose-master/data/0714/img/", help="dataset path dict")
    parser.add_argument('--cut_json_file', type=str, default="/home/guxiaowei/mmpose-master/data/0714/cut_img/", help="json file to train")
    parser.add_argument('--json_file', type=str, default="/home/guxiaowei/mmpose-master/data/0714/xml2json/", help="json path")
    
    args = parser.parse_args()
    xml2json(args.src_xml_file, args.cut_json_file, args.json_file)