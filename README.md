# attendance-system

Repository for the course project done as part of CS-337 (Artificial Intelligence & Machine Learning) course at IIT Bombay in Autumn 2022.  
Webpage: https://sarthakmittal92.github.io/projects/aut22/attendance-system/

## Abstract
This project is based on building an automated attendance system using face recognition. We make use of multiple models chained together to achieve this. Our framework consists of a pre-trained deep neural network Caffe model for face detection within an image, deployed using OpenCV DNN configuration, followed by alignment using Haar Cascade Eye Detector. This is followed by use of Inception-ResNet v1 architecture combined with Triplet Loss. We conclude by checking classification using Softmax.

## Code Structure
The [5-student-faces-dataset](5-student-faces-dataset/) folder contains training and validation data for the face classifier.
The [backbone](backbone/) folder has the pre-trained models, including DNN Caffe Model, Haar Cascade Classifier and Inception-ResNet.
The [files](files/) folder has text configuration files that are used to generate input for triplet loss that includes anchor, positive and negative sample.
The [images](images/) and [lfw](lfw/) folders have input images.
The [Notebooks](Notebooks/) folder has some helper code.
The [outputs](outputs/) folder has the pre-processed (face detected and aligned) images.

## Running
The [load_dataset](load_dataset.py) script first takes the input images and passes it through the detector (DNN Caffe Model) and aligner (Haar Cascade Classifier), and stores the processed data into [5-student-faces-dataset.npz](5-student-faces-dataset.npz).

The [compute_embeddings](compute_embeddings.py) script then takes the processed data archive and computes the embeddings by passing it through the feature extractor (Inception-ResNet) and stores the embeddings into [5-student-faces-embeddings.npz](5-student-faces-embeddings.npz).

Finally, there is a choice of taking one of the classifiers - [KNN](knn_classifier.py), [Softmax](softmax_classifier.py) or [SVM](svm_classifier.py), which will use the embeddings archive, validate the classification, and compute the accuracy. We achieved best results with the softmax classifier.