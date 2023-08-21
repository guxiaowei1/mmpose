import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import multiprocessing
import argparse

 
 
 
def convert(size, box):
 
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
 
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
 
 
    return (x, y, w, h)
 
 
def convert_annotation(xml_files_path, save_txt_files_path, classes, classes_all, cut, make_txt):
    classes_count = {'jyz':0, 'JYZ':0,'JYZ_XS':0,'JYZ_ZC':0}
    
    xml_ori_path = '/home/guxiaowei/mmpose-master/data/0714/'
    xml_files = os.listdir(xml_files_path)
    for xml_name in xml_files:
        if not (xml_name.endswith('.jpg') or xml_name.endswith('.png') or xml_name.endswith('.JPG')):
            continue
        if not os.path.exists(os.path.join(xml_ori_path, xml_name[:-4]+'.xml')):
            continue
        
        print(xml_name)
        count = 0
        xml_file = os.path.join(xml_ori_path, xml_name[:-4] + '.xml')
        out_txt_path = os.path.join(save_txt_files_path, xml_name[:-4] + '.txt')
        if make_txt:
            out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        print(os.path.join(xml_files_path, xml_name))
        if w == 0 or h == 0:
            imgtest = cv2.imread(os.path.join(xml_files_path, xml_name))
            w, h = imgtest.shape[1], imgtest.shape[0]
        if cut:
            imgtest = cv2.imread(os.path.join(xml_files_path, xml_name))

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                if cls not in classes_all:
                    # raise Exception("found error object name! Name is:", cls, "in file:", xml_file)
                    continue
                else:
                    continue
            count += 1
            cls_id = classes[cls]
            classes_count[cls] += 1
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text))
            if b[0]==b[1] or b[2]==b[3]:
                print(f'same cordinates in {b}')
                continue
            
            if cut:
                path_dir = os.path.join('/home/guxiaowei/mmpose-master/data/0714/cut_img/', cls)
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                new_path = path_dir + '/' + xml_name[:-4] + '_' + cls +str(count) + '.jpg'
                if w > 1 and h > 1:
                    if os.path.exists(new_path):
                        continue
                    x1, y1 = int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text)
                    x2, y2 = int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)
                    cut_w, cut_h = x2 - x1, y2 - y1
                    x1, y1 = max(0, int(x1 - cut_w / 5)), max(0, int(y1 - cut_h / 5))
                    x2, y2 = min(w, int(x2 + cut_w / 5)), min(h, int(y2 + cut_h / 5))
                    cv2.imwrite(new_path, imgtest[y1:y2, x1:x2, :])
            if make_txt:
                print(w, h, b)
                bb = convert((w, h), b)
                out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    print(classes_count)
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/guxiaowei/mmpose-master/data/0714/", help="dataset path dict")
    parser.add_argument('--save_txt_files', type=str, default="", help="json file to train")
    parser.add_argument('--cut', action='store_true', help="cut image")
    parser.add_argument('--make_txt', action='store_true', help="make txt label")
   
    classes_all = { 'JYZ':0,'jyz':0, 'JYZ_XS':0,'JYZ_ZC':0,'jyz_xs':0,'jyz_zc':0}
    
    classes =  { 'JYZ':0,'jyz':0, 'JYZ_XS':0,'JYZ_ZC':0,'jyz_xs':0,'jyz_zc':0}
    

    args = parser.parse_args()
    convert_annotation(args.data_path, args.save_txt_files, classes, classes_all, args.cut, args.make_txt)
 