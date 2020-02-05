from google.colab import drive
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

drive.mount('/content/drive/')
xy=np.loadtxt('/content/drive/My Drive/Colab Notebooks/Dataset/data-01-test-score.csv',delimiter=',', dtype=np.float32)

torch.manual_seed(777) #랜덤설정


#데이터셋 셔플

#x데이터와 y데이터 구분
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

#데이터 텐서로 변환
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#print(x_train)
#print(y_train)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=0.00005)


nb_epochs = 200
for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        #cost = F.binary_cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 10 == 0:
          print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
		  
		  
		  
		  
from google.colab import drive
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

drive.mount('/content/drive/')
xy=np.loadtxt('/content/drive/My Drive/Colab Notebooks/Dataset/data-01-test-score.csv',delimiter=',', dtype=np.float32)

torch.manual_seed(777) #랜덤설정


#데이터셋 셔플

#x데이터와 y데이터 구분
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

#데이터 텐서로 변환
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#정규화
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma

#print(norm_x_train)

#print(x_train)
#print(y_train)


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=0.05)

def train(model, optimizer, x_train, y_train):
  nb_epochs = 200
  for epoch in range(nb_epochs):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)
        #cost = F.binary_cross_entropy(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 10 == 0:
          print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

train(model, optimizer, norm_x_train, y_train)


