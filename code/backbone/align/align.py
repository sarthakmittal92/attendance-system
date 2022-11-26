# https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/

# To align the detected faces

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sys import argv
from os import listdir, mkdir
from tqdm import tqdm

# https://raw.githubusercontent.com/kipr/opencv/master/data/haarcascades/haarcascade_eye.xml

eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

try:
    mkdir(f'../../outputs/aligned')
except:
    pass

for dir in listdir(f'../../outputs/detected'):
    try:
        mkdir(f'../../outputs/aligned/{dir}')
    except:
        pass
    print("Aligning faces in", dir)
    for file in tqdm(listdir(f'../../outputs/detected/{dir}')):
    # reading image
        img = cv2.imread(f'../../outputs/detected/{dir}/{file}')
        img_raw = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        eyes = eye_detector.detectMultiScale(img_gray)

        index = 0
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            if index == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif index == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)
            cv2.rectangle(img,(eye_x, eye_y),(eye_x+eye_w, eye_y+eye_h), cv2.COLOR_BGR2GRAY, 2)
            index = index + 1

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]

        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]

        cv2.circle(img, left_eye_center, 2, (255, 0, 0) , 2)
        cv2.circle(img, right_eye_center, 2, (255, 0, 0) , 2)
        cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)

        # rotating the image on the basis of the angle of the eyeline
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse direction of clock

        cv2.circle(img, point_3rd, 2, (255, 0, 0) , 2)

        cv2.line(img,right_eye_center, left_eye_center,(67,67,67),2)
        cv2.line(img,left_eye_center, point_3rd,(67,67,67),2)
        cv2.line(img,right_eye_center, point_3rd,(67,67,67),2)

        def euclidean_distance(a, b):
            x1 = a[0]; y1 = a[1]
            x2 = b[0]; y2 = b[1]
            return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)

        cos_a = (b*b + c*c - a*a)/(2*b*c)

        angle = np.arccos(cos_a)

        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))
        
        try:
            cv2.imwrite(f'../../outputs/aligned/{dir}/aligned_{file.split("detected_")[1].split(".")[0]}.png',new_img)
        except:
            pass