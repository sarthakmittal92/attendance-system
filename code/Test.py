import sys
import os
sys.path.append("../")

import numpy as np
import cv2
import torch
import argparse
from backbone.networks.inception_resnet_v1 import InceptionResnetV1

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def find_id(output, Attendance_Database):
    representation = output
    minimum = 100
    identity = None
    for (name, db_enc) in Attendance_Database.items():
        dist = findCosineDistance(db_enc, representation)
        if dist < minimum:
            minimum = dist
            identity = name

    if minimum > 0.4:
        return None
    else:
        return identity , dist

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Face Verification Training')
    parser.add_argument( '--test', type= str, help='path to test dataset directory')

    args = parser.parse_args()
    Attendance_Database = {}
    Test_image = {}

    network = InceptionResnetV1(pretrained='vggface2')
    network.load_state_dict(torch.load("experiments/best_inception_resnet_V1_pretrained_triplet.pth", map_location=torch.device('cpu')))

    for folder in os.listdir(args.test):
        i = 0
        for image in os.listdir(args.test + folder):
            img = cv2.imread(args.test  + folder + "/" + image)
            img = cv2.resize(img, (160,160))

            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0)

            output = network(img)
            output = output.detach().cpu().numpy()
            output = output[0]
            if i == 0:
                Test_image[folder] = output
                i+=1
                continue
            
            if folder in Attendance_Database:
                Attendance_Database[folder].append(output)
            else:
                Attendance_Database[folder] = [output]
            i+=1

    for key in Attendance_Database:
        Attendance_Database[key] = np.array(Attendance_Database[key])
        Attendance_Database[key] = np.mean(Attendance_Database[key], axis=0)

    for folder, output in Test_image.items():

        id , _ = find_id(output, Attendance_Database)
        if id == None:
            print(folder ,": Unknown")
            continue

        if id == folder:
            print(folder ,": Correctly Identified as ", id)
        else:
            print(folder ,": Wrongly Identified as ", id)