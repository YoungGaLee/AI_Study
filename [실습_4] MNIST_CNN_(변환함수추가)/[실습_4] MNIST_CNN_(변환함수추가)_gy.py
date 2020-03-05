#코랩에서 돌아가지 않는 코드입니다... 어떤 에러인지도 안뜨고... 세션이 다운되고 비정상 종료됩니다...

#저번 MNIST_CNN_import : 이거 다 없어도 될거 같음
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.optim as optim
from torch.utils.data import DataLoader

#변환 코드_import 
import numpy as np
from torch.utils.data import Dataset
import torch
import time

#클래스 따로 위쪽으로 뺌.
  #변환코드클래스
class MyDataset(Dataset):
  
    def __init__(self, image_path, label_path):
        self.image_data = torch.from_numpy(self.read_image(image_path)) #numpy배열 > tensor로 바꿔줌
        self.label_data = torch.from_numpy(self.read_label(label_path)).long()
        self.len = self.label_data.size()[0]
        #__init__부분에서 read_image함수를 불러오고 read_image에서 read함수를 불어오기 때문에
        #따로 사용하는 것이 아니라 그냥 MyDataset에 경로지정해주면 알아서 진행

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.image_data[item].float() ,self.label_data[item]


    #read라는 함수를 돌릴때마다 앞의 4자리 읽어서 출력하는거 28 또는 0으로 되더라
    def read(self, data_file): # 앞의 4byte를 읽은 후 표준 형식으로 바꿔 출력    ????numpy????
        dt = np.dtype(np.uint32).newbyteorder('>')#np.dtype: ,numpy.dtype.newbyteorder(uint32)부호화되지 않은 32비트=4바이트 순서로 있는 원래 자료를 해석할 수 있도록 배열 dtype에 바이트 순서 정보를 변경하는 것
        # >u4 : 4글자 유니코드 문자열
        #what = np.frombuffer(data_file.read(4), dt)
        #print(what)
        #print(what[0])

        return np.frombuffer(data_file.read(4), dt)[0] #np.frombuffer( 바꾸고 싶은 bytes , dtype = <자료형>)

    def read_image(self, image_path):
        image_file = open(image_path, 'rb')
        # <_io.BufferedReader name='/content/drive/My Drive/MNIST_byte/t10k-images.idx3-ubyte'>

        image_file.read(4) # 처음 4byte는 데이터가 MNIST라는것을 의미
        # read()함수가 실행되는것이 아님....이거 뭐야? 그냥 네개를 읽어라.
        # read() : 파일 전체의 내용을 하나의 문자열로 읽어온다.그냥 내장함수인듯...??
        #b"\x00\x00'\x10"
        

        
        #계속 같은 ubyte파일을 넣어줌
        #앞에 이게 파일에서 얼마로 이루어졌다 하는 정보인듯함
        num_images = self.read(image_file) #10000 : 처음4개
        rows = self.read(image_file) # 28 : 그다음4개
        cols = self.read(image_file) # 28: 그다다음4개


        buf = image_file.read(rows * cols * num_images) # 28*28*10000개(전체 수) 읽기(불러오기) = buf:test_image파일에 있는 전체 내용을 담고 있음.
        #다 읽어왔으니까 close
        image_file.close()
        

        data = np.frombuffer(buf, np.uint8) # ?? 왜 다시 이걸로 바꾸지..? 1바이트로   ?????????
        data = data.reshape(num_images, 1, rows, cols) #data : (1000,1,28,28)로 reshape
        # print(data)#array
        return data

    def read_label(self, label_path):
        label_file = open(label_path, 'rb')
        label_file.read(4)# 처음 4byte는 데이터가 MNIST라는것을 의미(동일) / self없으니까 그냥 python 내장함수인 read()함수 사용
        num_label = self.read(label_file)#self.read니까 class에 있는 read함수 사용 /전체를 4byte로 바꿈?  ?????딱 4개만 바꾼거 아닌가????
        # print("여기 지나가긴 하니?")
        # print(num_label) #10000
        # print("이거야?")

        # print(type(label_file)) #buffer reader 처리 된거
        # print(type(num_label))


        buf = label_file.read(num_label) # 10000개 읽어옴.
        label_file.close()

        labels = np.frombuffer(buf, np.uint8)
        return labels



# 2개 레이어 CNN 클래스
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # Final FC 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x[:,0,:,:]
        w,h = x.shape[1],x.shape[2]
        x = x.view(-1,1,w,h)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc(out)
        return out
  
#------------------------------------------------------------------------------

#기본 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


#경로지정
Image_path_for_test = '/content/drive/My Drive/MNIST_byte/t10k-images.idx3-ubyte' # test_X
Label_path_for_test = '/content/drive/My Drive/MNIST_byte/t10k-labels.idx1-ubyte' # test_Y

image_path_for_train = '/content/drive/My Drive/MNIST_byte/train-images.idx3-ubyte' # train_X
Label_path_for_train = '/content/drive/My Drive/MNIST_byte/train-labels.idx1-ubyte' # train_Y


#클래스사용
dataset_for_test = MyDataset(Image_path_for_test,Label_path_for_test)
dataset_for_train = MyDataset(image_path_for_train,Label_path_for_train)
model = CNN().to(device)


# parameters
train_X = dataset_for_train.image_data
train_Y = dataset_for_train.label_data

test_X = dataset_for_test.image_data
test_Y = dataset_for_train.label_data

learning_rate = 0.001
epochs = 10
# batch_size = 200 #ubyte를 바꾸는거니까 용량상관없으니 안써도 되지 않을까?
# num_batches = len(train_Y) // batch_size


# loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training
total_batch = len(train_Y) # 보류
print('Learning started. It takes sometime.')
num = 0
for epoch in range(epochs+1): # epoch먼저
    avg_cost = 0
    num += 1
    # image is already size of (28x28), no reshape
    # label is not one-hot encoded
    X = train_X.to(device)
    Y = train_Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch
    print(num) #180번대부터 속도 확 꺽임

    if num % 100 == 0:
        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


print('Learning Finished!')

# Test model and check accuracy
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

