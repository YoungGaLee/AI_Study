# 1. 필요한 모듈 선언

from google.colab import drive
import torch
import torch as nn
import torch.nn.functional as F
import torch.optim as optim 
import pandas as pd
import numpy as np

drive.mount('/content/drive/')
data=np.genfromtxt('/content/drive/My Drive/dataset/data-01-test-score.csv', delimiter=',', dtype=np.float32)


# 2. 시드 고정
torch.manual_seed(1) #랜덤설정


# 3. 데이터셋 로드 & 전처리

#데이터
x_train = torch.FloatTensor(data[:,:3]) #25x3 >>W:3x1 // b:25X1
y_train = torch.FloatTensor(data[:,-1]) #25

Y_train = y_train.unsqueeze(1)#25x1 :사실 사이즈변형 안해줘도 될지도


# 모델 초기화
W = torch.zeros((3,1),requires_grad=True) # 1x1:이래도 돼?  torch.zeros((3,1))해야 하는거 아닌가
b = torch.zeros((25,1),requires_grad=True) # 0. :초기화를 0으로 잡고 돌려보겠다.

'''
print(x_train.shape)
print(W.shape)
#print(b.shape)
hypothesis = x_train.matmul(W) 
print(hypothesis.shape)
'''

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.00005005)
#SGD vs GD : GD는 전체를 다보고 레이트조정, SGD는 mini-batch만 보고 조정(발자국 작음)


#learning_rate = 0.01 #SGD버전이랑 같게 하려고
nb_epochs = 1000 
for epoch in range(nb_epochs + 1): # 1001번 돌리기 == update를 1001번해주기 같은 데이터로(나눠서 올리는게 아닌가봄) == 1001번 학습한다?

    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # [25X3]*[3x1]+[1x1] = [25x1] 인걸 보면 알아서 맞춰지는 듯
    #print(hypothesis)
    
    #print(Y_train)
    #print(hypothesis - Y_train)
    

    # cost 계산
    cost = torch.mean((hypothesis - Y_train) ** 2)
    #print(cost)

    # cost로 H(x) 개선
    optimizer.zero_grad() #초기화 매학습에 0으로 초기화 뭐를?
    cost.backward()
    optimizer.step() # rate 조정된 다음 W로

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs,cost.item()))
