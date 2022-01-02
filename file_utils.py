# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import pytesseract
import imutils
import math
from collections import deque
from utils.code_image_cleaner import process_image_for_ocr
from utils.utilss import display_image_cv2, resize_to_suitable
from utils import ocr




def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def rect_distance(x11, y11, x12, y12, x21, y21, x22, y22):
    left = y22 < y11
    right = y12 < y21
    bottom = x12 < x21
    top = x11 > x22
    # if left: print("left")
    # if right: print("right")
    # if bottom: print("bottom")
    # if top: print("top")


    if top and left:
        return dist(x11, y11, x22, y22)
    elif left and bottom:
        return dist(x12, y11, x21, y22)
    elif bottom and right:
        return dist(x12, y12, x21, y21)
    elif right and top:
        return dist(x11, y12, x22, y21)
    elif left:
        return y11 - y22
    elif right:
        return y21 - y12
    elif bottom:
        return x21 - x12
    elif top:
        return x11 - x22
    else:             
        return 0


pytesseract.pytesseract.tesseract_cmd = r'F:\Program files\Tesseract-OCR\tesseract.exe'


def check_number(x):
    return x.isdigit()

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

# Duyệt từng ảnh trong 1 folder và phân loại
def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


# Lưu kết quả: ảnh, file txt và vẽ bounding box cho vùng chứa ký tự
def saveResult(img_file, img, root_image, boxes, r, dirname='./result/', verticals=None, texts=None, position_crop=(0,0)):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1

        img = np.array(img)
        root_image = np.array(root_image)

        p1, p2 = position_crop

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        
        img1 = img.copy()
        
        h = np.asarray(boxes).shape[0]
        list = [-1] * h #Đánh dấu các box gần nhau
        b = []  # Chứa kích thước của các box lớn
        orientation_box = [] # = 0-> horizon
                             # = 1-> vertical
        # center_box = [] # Lấy điểm trung tâm của từng box

        # print(boxes.shape)

        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)

                # print(poly)
                x1 = min(poly[1], poly[3])
                x2 = max(poly[5], poly[7])
                y1 = min(poly[0], poly[6])
                y2 = max(poly[2], poly[4])
                b.append([x1,y1,x2,y2])

            
                # Gom các box gần nhau
                for ii, boxx in enumerate(boxes):
                    # print(f"ii = {ii}")
                    if ii > i:
                        polyy = np.array(boxx).astype(np.int32).reshape((-1))
                        x11 = min(polyy[1], polyy[3])
                        x22 = max(polyy[5], polyy[7])
                        y11 = min(polyy[0], polyy[6])
                        y22 = max(polyy[2], polyy[4])
                        dist = rect_distance(x1,y1,x2,y2,x11,y11,x22,y22)
                        if dist < 35:
                            if list[i] == -1:
                                list[ii] = i
                                # print(f"list[{ii}]={i}")
                            else:
                                list[ii] = list[i]
                                # print(f"list[{ii}] = {list[ii]}")
                            

                

                poly = poly.reshape(-1, 2)
                # cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                
                # cv2.putText(img, "{}".format(text[:-2]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                # print(text)
                # texts.append(text[:-2])

                # if texts is not None:
                #     cv2.putText(img, "{}".format(text), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                #     cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # print("-------")
        # Tính bounding box lớn của các box gần nhau dựa vào list (chứa thông tin các box gần nhau)
        for i,listt in enumerate(list):
            if listt != -1:
                x1,y1,x2,y2 = b[listt]
                x11,y11,x22,y22 = b[i]
                b[listt] = [min(x1,x11), min(y1,y11), max(x2,x22), max(y2,y22)]
                b[i] = [0,0,0,0]
        
        # print(b)
        # Hiển thị kết quả: vẽ bounding box lớn, hiển thị chữ nhận diện được
        for i, box in enumerate(b):
            if box[0] != 0:
                # print(i)
                w = box[3] - box[1]
                h = box[2] - box[0]
                # print(f"w = {w}, h = {h}")
                if w > h:
                    # print('ngang')
                    x = int(w*0.13)
                    y = int(h*0.13)
                    img_crop = img[(box[0]-3):(box[2]+y), (box[1]-3):(box[3] + 3 + x)].copy()
                    # cv2.imwrite(f"img{i}_original.jpg", img_crop)

                    # Clean image
                    if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
                        continue
                    img_crop = resize_to_suitable(img_crop)
                    clean_img = process_image_for_ocr(img_crop)
                    clean_img = cv2.bitwise_not(clean_img)
                    clean_img = cv2.blur(clean_img, (2, 2))

                    # cv2.imwrite(f"img{i}.jpg", clean_img)


                    # text = pytesseract.image_to_string(clean_img)
                    text = ocr.find_code_in_image(clean_img)


                    if (len(text) > 10):
                        if (not check_number(text[0])) and (not check_number(text[1])) and (not check_number(text[2])) and (not check_number(text[3])) and (check_number(text[4])) and (check_number(text[5])) and (check_number(text[6])) and (check_number(text[7])):
                            print("Code can be Container Code:")
                            print(text)
                            
                    cv2.putText(img, "{}".format(text), (box[1],box[0]-10), font, font_scale, (255, 255, 0), thickness=2)
                    cv2.rectangle(img, (box[1]-3,box[0]-3), (box[3]+3+x, box[2]+y), (0, 255, 0), 2)

                    cv2.putText(root_image, "{}".format(text), (p1+int(box[1]/r),p2+int((box[0]-10)/r)), font, font_scale, (255, 255, 0), thickness=2)
                    cv2.rectangle(root_image, (p1+int((box[1]-3)/r),p2+int((box[0]-3)/r)), (p1+int((box[3]+3+x)/r), p2+int((box[2]+y)/r)), (0, 255, 0), 2)

                            

                else:
                    # print('doc')
                    x = int(h*0.1)
                    img_crop = img[(box[0]-3):(box[2]+3 + x), (box[1]-3):(box[3] + 3)].copy()
                    # cv2.imwrite(f"img{i}_original.jpg", img_crop)

                    # Clean image
                    if img_crop.shape[0] == 0 or img_crop.shape[1] == 0:
                        continue
                    img_crop = resize_to_suitable(img_crop)
                    clean_img = process_image_for_ocr(img_crop)
                    clean_img = cv2.bitwise_not(clean_img)
                    clean_img = cv2.blur(clean_img, (2, 2))

                    # cv2.imwrite(f"img{i}.jpg", clean_img)


                    # text = pytesseract.image_to_string(img_crop)
                    text = ocr.find_code_in_image(clean_img)

                    if (len(text) > 10):
                        if (not check_number(text[0])) and (not check_number(text[1])) and (not check_number(text[2])) and (not check_number(text[3])) and (check_number(text[4])) and (check_number(text[5])) and (check_number(text[6])) and (check_number(text[7])):
                            print("Code can be Container Code:")

                            print(text)

                    cv2.putText(img, "{}".format(text), (box[1],box[0]-10), font, font_scale, (255, 255, 0), thickness=2)
                    cv2.rectangle(img, (box[1]-3,box[0]-3), (box[3]+3, box[2]+3), (0, 255, 0), 2)

                    cv2.putText(root_image, "{}".format(text), (p1+int(box[1]/r),p2+int((box[0]-10)/r)), font, font_scale, (255, 255, 0), thickness=2)
                    cv2.rectangle(root_image, (p1+int((box[1]-3)/r),p2+int((box[0]-3)/r)), (p1+int((box[3]+3)/r), p2+int((box[2])/r)), (0, 255, 0), 2)
                   
               

        # Save result image
        cv2.imwrite(res_img_file, img)
        # print("Write imc.png")
        cv2.imwrite('imc.png', root_image)

