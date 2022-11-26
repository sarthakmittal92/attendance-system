import numpy as np
from backbone.networks.inception_resnet_v1 import InceptionResnetV1
import torch

def get_embedding(model, face_pixels):

    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)

    samples = np.transpose(samples, (0,3,1,2))
    samples = torch.from_numpy(samples).float()

    yhat = model(samples)

    return yhat[0]

data = np.load('5-student-faces-dataset.npz')
x_train, y_train, x_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = InceptionResnetV1(pretrained='vggface2')
model.load_state_dict(torch.load("experiments/group22.pt", map_location=torch.device('cpu')))

print('Loaded Model')

newx_train = list()
for face_pixels in x_train:
	embedding = get_embedding(model, face_pixels)
	newx_train.append(embedding.detach().cpu().numpy())
newx_train = np.asarray(newx_train)
print(newx_train.shape)

newx_test = list()
for face_pixels in x_test:
	embedding = get_embedding(model, face_pixels)
	newx_test.append(embedding.detach().cpu().numpy())
newx_test = np.asarray(newx_test)
print(newx_test.shape)

np.savez_compressed('5-student-faces-embeddings.npz', newx_train, y_train, newx_test, y_test)