import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

torch.manual_seed(1)
np.random.seed(42)


dataset_path = './인공지능 두뇌지수 데이터_1024.xlsx'


data_select = ['Scholarly Output','Most recent publication','Citations','Citations per Publication','Field-Weighted Citation Impact','h-index','Country Number','Scholarly Output100','Citations per Publication100','Field-Weighted Citation Impact100','Degree']


dataset = pd.read_excel(dataset_path, usecols = data_select, index_col=0)



DATA = dataset.values
np.random.shuffle(DATA)


# dataset = pd.read_excel(dataset_path, sheet_name='Sheet1', header = 1, index_col=None, usecols=data_select)
# dataset = np.genfromtxt(dataset_path, names = (*data_select), 'header = 1' index_col=None)

x_data = DATA[:, 0:-1]
y_data = DATA[:, [-1]]


x_data = torch.FloatTensor(X_data)
y_data = torch.FloatTensor(y_data)

#print(x_train)
validation = len(x_data)/90 #for train


x_train = x_data[:validation + 1, :]
y_train = y_data[:validation + 1]

x_valid = x_data[validation+1:, :]
y_valid = , y_data[[validation+1:]


linear1 = torch.nn.Linear(10, 256, bias=True)
linear2 = torch.nn.Linear(256, 25, bias=True)

ReLu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, linear2, ReLU)

'''
##linear 
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear_1 = nn.Linear(10, 256, bias=True)
      self.linear_2 = nn.Linear(256, 25.bias=True)

    def forward(self, x):
      
        return self.linear(x)

'''
# optimizer 설정
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산 (2)
    z = x_train.matmul(W) + b 
    cost = F.cross_entropy(z, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

  '''
