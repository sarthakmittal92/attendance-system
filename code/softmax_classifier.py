import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

model = torch.nn.Sequential(
    torch.nn.Linear(512,1024, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(1024,128, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(128,5, bias=True),
    torch.nn.ReLU(),
    torch.nn.Softmax(dim=1)
)

torch.manual_seed(0)
print(model)

data = np.load('5-student-faces-embeddings.npz')
x_train, y_train, x_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (x_train.shape[0], x_test.shape[0]))

y_train = torch.tensor(np.array(pd.get_dummies(y_train)))
y_test = torch.tensor(np.array(pd.get_dummies(y_test)))

num_epoch = 1000

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)

x_train = torch.from_numpy(x_train).float()
y_train = y_train.float()
for epoch in range(num_epoch):

    input = Variable(x_train)
    target = Variable(y_train)

    # forward
    out = model(input)
    loss = loss_function(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # show
    print('Epoch[{}/{}], loss: {:.6f}'
          .format(epoch + 1, num_epoch, loss.data.item()))

# predicting
# print(type(x_train))
y_pred_train = model(x_train)
y_pred_test = model(Variable(torch.from_numpy(x_test).float()))

score_train = torch.sum(torch.max(y_train, 1)[1] == torch.max(y_pred_train, 1)[1]).item() / len(y_train)
score_test = torch.sum(torch.max(y_test, 1)[1] == torch.max(y_pred_test, 1)[1]).item() / len(y_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
