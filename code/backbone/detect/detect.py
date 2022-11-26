# https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/

# To detect and return faces from images

# model structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
# pre-trained weights: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sys import argv
from os import listdir, mkdir
from tqdm import tqdm

detector = cv2.dnn.readNetFromCaffe("deploy.prototxt" , "res10_300x300_ssd_iter_140000.caffemodel")
idx = 0

try:
    mkdir(f'../../outputs')
except:
    pass

try:
    mkdir(f'../../outputs/detected')
except:
    pass

for dir in listdir(f'../../{argv[1]}'):
    try:
        mkdir(f'../../outputs/detected/{dir}')
    except:
        pass
    print("Detecting faces in", dir)
    for file in tqdm(listdir(f'../../{argv[1]}/{dir}')):
        # image resizing
        image = cv2.imread(f'../../{argv[1]}/{dir}/{file}')
        base_img = image.copy()
        original_size = base_img.shape
        target_size = (300, 300)
        image = cv2.resize(image, target_size)
        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])

        # detector expects (1, 3, 300, 300) shaped input
        imageBlob = cv2.dnn.blobFromImage(image = image)

        # feed forward
        detector.setInput(imageBlob)
        detections = detector.forward()

        column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        detections_df = pd.DataFrame(detections[0][0], columns = column_labels)

        # 0: background, 1: face
        detections_df = detections_df[detections_df['is_face'] == 1]
        detections_df = detections_df[detections_df['confidence'] >= 0.90]

        detections_df['left'] = (detections_df['left'] * 300).astype(int)
        detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
        detections_df['right'] = (detections_df['right'] * 300).astype(int)
        detections_df['top'] = (detections_df['top'] * 300).astype(int)

        for i, instance in detections_df.iterrows():
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            left = instance["left"]; right = instance["right"]
            bottom = instance["bottom"]; top = instance["top"]
            detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y) ,
            int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
            # print(idx) 
            idx += 1
            try:
                cv2.imwrite(f'../../outputs/detected/{dir}/detected_{file.split(".")[0]}.png',detected_face[:,:,::-1])
            except:
                pass