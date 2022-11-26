import sys
import os
sys.path.append("../")

import numpy as np
import cv2
import torch
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

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
        dist = findEuclideanDistance(db_enc, representation)
        if dist < minimum:
            minimum = dist
            identity = name

    return identity

if __name__ == '__main__':

    data = load('5-student-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    Attendance_Database = {}
    counts = {}
    for i in range(len(trainy)):

        if trainy[i] not in Attendance_Database:
            Attendance_Database[trainy[i]] = trainX[i]
            counts[trainy[i]] = 1
        else:
            Attendance_Database[trainy[i]] += trainX[i]
            counts[trainy[i]] += 1
    
    Attendance_Database = {k: v / counts[k] for k, v in Attendance_Database.items()}

    correct = 0
    for idx , emd in enumerate(testX):
        id = find_id(emd, Attendance_Database)
        if id == trainy[idx]:
            correct += 1
    
    print("Accuracy: ", correct/len(testy))
