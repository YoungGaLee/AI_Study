https://colab.research.google.com/drive/1oug1HDcmbRMdyleKoBPuyKsOKWrlv6fz


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#시드고정
torch.manual_seed(1)
np.random.seed(777)

#파라미터
learning_rate = 0.001
nb_epochs = 1000

#데이터로드 및 셔플
al_data = pd.read_excel('/content/drive/My Drive/인공지능 두뇌지수 데이터_1024.xlsx', usecols=['Scholarly Output','Most recent publication','Citations','Citations per Publication','Field-Weighted Citation Impact','h-index','Country Number','Scholarly Output100','Citations per Publication100','Field-Weighted Citation Impact100','Degree'])
xy_data = al_data.values

np.random.shuffle(xy_data)

#print(al_data)
#print(xy_data)

#데이터 구분
x_data = xy_data[:,0:-1]
y_data = xy_data[:,[-1]]

#데이터 텐서로 변환
x_train = torch.FloatTensor(x_data)
y_train = torch.LongTensor(y_data)


#x 데이터 값 정규화
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma

#print(norm_x_train)
print(norm_x_train.shape)
print(y_train.shape)

#x_data.dtype
#y_data.dtype

#검증데이터 테스트데이터 구분


#모델생성

linear1 = torch.nn.Linear(10, 256, bias=True)
linear2 = torch.nn.Linear(256, 1, bias=True)
relu = torch.nn.ReLU()

model = torch.nn.Sequential(linear1, relu, linear2)
#print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(nb_epochs + 1):
    hypothesis = model(x_train)

    #cost = F.cross_entropy(hypothesis, y_train)
    #cost = torch.nn.CrossEntropyLoss(hypothesis, y_train)
    cost = F.mse_loss(hypothesis, y_train)


    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
