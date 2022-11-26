import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

data = np.load('5-student-faces-embeddings.npz')
x_train, y_train, x_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (x_train.shape[0], x_test.shape[0]))

in_encoder = Normalizer(norm='l2')
x_train = in_encoder.transform(x_train)
x_test = in_encoder.transform(x_test)

out_encoder = LabelEncoder()
out_encoder.fit(y_train)
y_train = out_encoder.transform(y_train)
y_test = out_encoder.transform(y_test)

model = SVC(kernel='linear', probability=True)
model.fit(x_train, y_train)

yhat_train = model.predict(x_train)
yhat_test = model.predict(x_test)

score_train = accuracy_score(y_train, yhat_train)
score_test = accuracy_score(y_test, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))