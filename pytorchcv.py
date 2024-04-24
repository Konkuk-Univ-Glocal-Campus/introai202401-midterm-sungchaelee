import builtins
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_mnist(batch_size=64):
    builtins.data_train = torchvision.datasets.FashionMNIST('./data',
        download=True, train=True, transform=ToTensor())  # Fashion MNIST 학습 데이터셋 로드
    builtins.data_test = torchvision.datasets.FashionMNIST('./data', 
        download=True, train=False, transform=ToTensor())  # Fashion MNIST 테스트 데이터셋 로드
    builtins.train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)  # 학습 데이터 로더
    builtins.test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)  # 테스트 데이터 로더

def train_epoch(net, dataloader, lr=0.01, optimizer=None, loss_fn=nn.CrossEntropyLoss(), device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    net.to(device)
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()  # 모델을 학습 모드로 설정

    total_loss, acc, count = 0, 0, 0
    for features, labels in dataloader:
        optimizer.zero_grad()  # 최적화 도구의 모든 기울기를 0으로 초기화
        features, labels = features.to(device), labels.to(device)  # 데이터를 기본 계산 장치로 이동
        
        outputs = net(features)  # 신경망을 통해 예측 수행
        loss = loss_fn(outputs, labels)  # 손실 계산
        loss.backward()  # 손실에 대한 기울기 계산
        optimizer.step()  # 계산된 기울기를 이용해 신경망의 가중치를 업데이트
        
        total_loss += loss.item()  # 총 손실을 누적
        _, predicted = torch.max(outputs, 1)  # 예측된 결과 중 가장 높은 확률을 가진 클래스를 선택
        acc += (predicted == labels).sum().item()  # 정확하게 예측된 수를 누적
        count += labels.size(0)  # 처리된 레이블의 수를 누적

    return total_loss / count, acc / count  # 평균 손실과 정확도 반환

def validate(net, dataloader, loss_fn=nn.NLLLoss()):
    net.eval()  # 모델을 평가(evaluation) 모드로 설정
    total_loss, total_correct, total_samples = 0, 0, 0  # 초기화
    with torch.no_grad():  # 기울기 계산을 비활성화하여 메모리 소비를 줄이고 계산 속도를 향상
        for features, labels in dataloader:  # 데이터 로더를 통해 데이터를 배치 단위로 받아옴
            labels = labels.to(default_device)  # 레이블을 기본 계산 장치로 이동
            outputs = net(features.to(default_device))  # 특징 데이터도 계산 장치로 이동 후 모델에 통과
            batch_loss = loss_fn(outputs, labels)  # 손실 계산
            total_loss += batch_loss.item() * features.size(0)  # 배치 손실을 전체 손실에 추가 (손실은 평균이므로 샘플 수를 곱해줌)

            _, predictions = torch.max(outputs, 1)  # 예측 결과를 가져옴
            total_correct += (predictions == labels).sum().item()  # 정확한 예측 수를 누적
            total_samples += labels.size(0)  # 처리한 총 샘플 수를 누적

    average_loss = total_loss / total_samples  # 평균 손실 계산
    accuracy = total_correct / total_samples  # 정확도 계산
    return average_loss, accuracy

def train(net, train_loader, test_loader, optimizer=None, lr=0.01, epochs=10, loss_fn=nn.NLLLoss()):
    # 최적화 도구가 제공되지 않은 경우 Adam 옵티마이저 사용, 학습률은 lr로 설정
    optimizer = optimizer or optim.Adam(net.parameters(), lr=lr)
    
    # 결과를 저장할 딕셔너리 초기화
    results = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for ep in range(epochs):  # 지정된 에폭 수만큼 반복
        # 한 에폭 동안 학습을 수행하고, 학습 손실과 정확도를 반환
        train_loss, train_acc = train_epoch(net, train_loader, optimizer=optimizer, loss_fn=loss_fn)
        # 모델을 검증 데이터셋에 대해 평가하고, 검증 손실과 정확도를 반환
        val_loss, val_acc = validate(net, test_loader, loss_fn=loss_fn)
        
        # 에폭, 학습 정확도, 검증 정확도, 학습 손실, 검증 손실을 출력
        print(f"Epoch {ep:2}, Train acc={train_acc:.3f}, Val acc={val_acc:.3f}, Train loss={train_loss:.3f}, Val loss={val_loss:.3f}")
        
        # 각 결과 값을 딕셔너리에 추가
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
    
    return results  # 학습과 검증 과정에서의 결과를 담은 딕셔너리를 반환

def train_long(net, train_loader, test_loader, epochs=5, lr=0.01, optimizer=None, loss_fn=nn.NLLLoss(), print_freq=10):
    # 최적화 도구가 제공되지 않은 경우 Adam 옵티마이저 사용, 학습률은 lr로 설정
    optimizer = optimizer or optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):  # 지정된 횟수만큼 에폭을 반복
        net.train()  # 모델을 학습 모드로 설정
        total_loss, correct, count = 0, 0, 0  # 총 손실, 정확도 계산을 위한 누적된 정확도, 그리고 처리된 데이터 수를 초기화
        
        for i, (features, labels) in enumerate(train_loader):  # 학습 데이터 로더로부터 데이터 배치를 가져옴
            labels = labels.to(default_device)  # 레이블을 기본 계산 장치로 이동
            optimizer.zero_grad()  # 기울기 버퍼를 0으로 초기화
            outputs = net(features.to(default_device))  # 특징 데이터를 기본 계산 장치로 이동 후 모델에 통과
            loss = loss_fn(outputs, labels)  # 손실 계산
            loss.backward()  # 손실에 대한 기울기 계산
            optimizer.step()  # 계산된 기울기를 사용하여 모델 가중치를 갱신
            total_loss += loss.item() * features.size(0)  # 총 손실을 누적
            _, predicted = torch.max(outputs, 1)  # 예측된 결과 중 가장 높은 값을 가진 클래스의 인덱스 추출
            correct += (predicted == labels).sum().item()  # 정확한 예측 수를 누적
            count += labels.size(0)  # 처리된 데이터 수를 누적
            
            if i % print_freq == 0:  # 지정된 빈도마다 현재까지의 학습 상태 출력
                print(f"Epoch {epoch}, minibatch {i}: train acc = {correct / count}, train loss = {total_loss / count}")
        
        # 한 에폭이 끝날 때마다 검증 함수를 호출하여 검증 손실과 정확도 계산
        val_loss, val_acc = validate(net, test_loader, loss_fn)
        print(f"Epoch {epoch} done, validation acc = {val_acc}, validation loss = {val_loss}")

def plot_results(hist): # plot_results라는 함수를 정의하는데 hist라는 이름의 딕셔너리를 매개변수로 받는데 학습과 검증 과정의 정확도와 손실이 배열 형태로 저장되어 있음
    plt.figure(figsize=(15,5)) # 새로운 그래프 창을 만들고, 크기를 가로 15인치, 세로 5인치로 설정
    plt.subplot(121) # 두 개의 그래프를 나란히 표시하기 위해 첫 번째 위치(1행 2열의 첫 번째)에 서브플롯을 생성
    plt.plot(hist['train_acc'], label='Training acc') # hist 딕셔너리에서 학습 정확도(train_acc)를 추출하여 그래프로 그리는데 라벨을 'Training acc'로 지정하여 그래프에 범례를 추가
    plt.plot(hist['val_acc'], label='Validation test') # hist 딕셔너리에서 검증 정확도(val_acc)를 추출하여 그래프로 그리는데 라벨을 'Validation acc'로 지정
    plt.legend() # 그래프에 범례를 추가하는데 각 데이터 세트를 구분하기 위해 사용
    plt.subplot(122) # 두 번째 위치(1행 2열의 두 번째)에 또 다른 서브플롯을 생성
    plt.plot(hist['train_loss'], label='Training loss') # hist 딕셔너리에서 학습 손실(train_loss)을 추출하여 그래프로 그린는데 라벨을 'Training loss'로 지정
    plt.plot(hist['val_loss'], label='Validation loss') # hist 딕셔너리에서 검증 손실(val_loss)을 추출하여 그래프로 그리는데 라벨을 'Validation loss'로 지정
    plt.legend() # 그래프에 범례를 추가

def plot_convolution(t, title=''): # 함수를 정의하고, 두 개의 매개변수를 받는데 t: 컨볼루션 연산에 사용될 커널의 텐서이고 title: 그래프의 상단에 표시될 제목인데 기본값은 빈 문자열
    with torch.no_grad(): # 이 블록 내에서는 PyTorch의 자동 미분 기능을 비활성화하여, 연산에 대한 기울기 계산을 수행하지 않는데 메모리 사용을 줄이고 연산 속도를 향상
        c = nn.Conv2d(kernel_size=(3,3), out_channels=1, in_channels=1) # 3x3 크기의 커널을 사용하는 2D 컨볼루션 레이어를 생성하는데 입력 채널과 출력 채널이 모두 1
        c.weight.copy_(t.unsqueeze(0)) # 입력된 텐서 t를 컨볼루션 레이어의 가중치로 복사하는데 이렇게 설정하면 컨볼루션 연산이 정확히 이 가중치를 사용
        fig, ax = plt.subplots(2, 6, figsize=(8, 3)) # 2행 6열의 서브플롯을 생성하는데 그래프의 전체 크기는 가로 8인치, 세로 3인치
        fig.suptitle(title, fontsize=16) # 전체 그래프의 제목을 설정하는데 폰트 크기는 16
        for i in range(5): # 5번 반복하여 첫 번째 5개의 이미지에 대해 다음 작업을 수행
            im = data_train[i][0] # 학습 데이터셋에서 i번째 이미지를 호출
            ax[0][i].imshow(im.squeeze(), cmap='gray') # 첫 번째 행의 i번째 서브플롯에 원본 이미지를 흑백으로 표시
            ax[1][i].imshow(c(im.unsqueeze(0)).squeeze(), cmap='gray') # 두 번째 행의 i번째 서브플롯에 컨볼루션 결과를 흑백으로 표시
            ax[0][i].axis('off') # 각 서브플롯의 축을 숨김
            ax[1][i].axis('off')
        ax[0, 5].imshow(t.squeeze(), cmap='gray') # 첫 번째 행의 6번째 서브플롯에 커널 텐서를 흑백으로 표시하는데 이 이미지는 컨볼루션 연산의 기본이 되는 커널을 시각화
        ax[0, 5].axis('off') # 마지막 열의 축을 숨김
        ax[1, 5].axis('off')
        plt.show() # 그래프를 표시

def display_dataset(dataset, n=10, classes=None): # display_dataset 함수를 정의하며, 매개변수로는 dataset (이미지와 레이블을 포함하는 데이터셋), n (표시할 이미지 수, 기본값은 10), classes (클래스 레이블 이름 배열, 선택적)를 받음
    fig, ax = plt.subplots(1, n, figsize=(15, 3)) # 1행 n열의 서브플롯을 생성하고, 전체 그래프의 크기를 가로 15인치, 세로 3인치로 설정
    mn = min([dataset[i][0].min() for i in range(n)]) # 데이터셋에서 선택된 이미지들 중 픽셀 값의 최소값을 계산하는데 값은 이미지 정규화에 사용
    mx = max([dataset[i][0].max() for i in range(n)]) # 데이터셋에서 선택된 이미지들 중 픽셀 값의 최대값을 계산하는데 값은 이미지 정규화에 사용
    for i in range(n): # 0부터 n-1까지의 인덱스에 대해 반복
        ax[i].imshow(dataset[i][0][0], cmap='gray') # 각 이미지를 흑백으로 표시
        ax[i].axis('off') # 각 서브플롯의 축을 숨김
        if classes: # classes 매개변수가 제공되었을 경우, 각 이미지 위에 해당하는 클래스 레이블을 제목으로 설정하는데 dataset[i][1]은 i번째 이미지의 클래스 인덱스
            ax[i].set_title(classes[dataset[i][1]])

def check_image(fn): # check_image라는 이름의 함수를 정의하며, 매개변수로 파일 이름 fn을 받음
    try: # 예외 처리를 시작하는 블록인데 안에서 이미지 파일을 열고 검증을 시도
        im = Image.open(fn) # Image.open(fn)을 사용하여 파일 fn을 열고, im 객체에 할당하는데 여기서 Image는 파이썬의 PIL(Pillow) 라이브러리에서 제공하는 모듈이고 이 함수는 이미지 파일을 불러오는 데 사용
        im.verify() # im.verify() 메소드를 호출하여 이미지 데이터가 손상되었는지 또는 파일이 손상된 이미지 포맷을 포함하고 있는지 검증하는데 파일을 읽을 때 발생할 수 있는 다양한 예외를 감지할 수 있고 이미지 파일이 유효하다면 이 부분은 문제없이 실행
        return True # try 블록이 성공적으로 완료되면, True를 반환하는데 파일이 유효한 이미지라는 것을 의미
    except: # try 블록에서 예외가 발생하면 실행되는 블록인데 예외가 발생하는 경우, 이미지 파일이 유효하지 않거나 파일 열기 과정에서 문제가 발생한 것
        return False # 예외가 발생했으므로, False를 반환하는데 이는 파일이 유효한 이미지가 아니라는 것을 나타냄
    
def check_image_dir(path): # check_image_dir라는 이름의 함수를 정의하며, 매개변수로 경로 패턴 path를 받는데 경로는 검색할 이미지 파일의 위치를 지정하는 글로브 패턴(glob pattern)을 포함할 수 있음
    for fn in glob.glob(path): # 함수를 사용하여 주어진 경로 패턴과 일치하는 모든 파일을 찾아 반복하는데 glob.glob은 지정된 패턴과 일치하는 모든 경로명을 리스트로 반환하는 함수
        if not check_image(fn): # check_image(fn) 함수를 호출하여 현재 파일 fn이 유효한 이미지인지 검사하는데 check_image 함수는 파일이 이미지로서 유효하지 않을 경우 False를 반환
            print("Corrupt image: {}".format(fn)) # 파일이 손상되었다면 해당 파일 이름과 함께 "Corrupt image" 메시지를 출력
            os.remove(fn) # os.remove(fn) 함수를 호출하여 손상된 이미지 파일을 파일 시스템에서 삭제

def common_transform(): # common_transform이라는 이름의 함수를 정의하는데 이 함수는 매개변수를 받지 않고, 구성된 이미지 변환 파이프라인을 반환
    std_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]) # torchvision.transforms.Normalize 함수를 사용하여 이미지의 각 채널에 대한 정규화를 설정하는데 이 변환은 주어진 평균(mean)과 표준편차(std)를 사용하여 각 채널의 픽셀 값을 정규화하는데 이 값들은 일반적으로 ImageNet 데이터셋을 기준으로 한 통계값
    trans = torchvision.transforms.Compose([ # 여러 이미지 변환 단계를 조합하는 torchvision.transforms.Compose 함수를 사용
            torchvision.transforms.Resize(256), # 이미지의 크기를 256x256 픽셀로 조정
            torchvision.transforms.CenterCrop(224), # 이미지의 중앙을 기준으로 224x224 픽셀의 크기로 중앙을 자릅
            torchvision.transforms.ToTensor(), # 이미지 데이터를 PyTorch 텐서로 변환하고, 데이터 타입을 0에서 1 사이의 값으로 스케일링
            std_normalize]) # 정규화 변환을 적용
    return trans # 구성된 변환 파이프라인을 반환

def load_cats_dogs_dataset(): # load_cats_dogs_dataset라는 이름의 함수를 정의합니다. 이 함수는 매개변수를 받지 않음
    if not os.path.exists('data/PetImages'): # 지정된 경로에 'PetImages' 폴더가 존재하는지 확인합니다. 폴더가 없으면 다음 단계로 이동
        with zipfile.ZipFile('data/kagglecatsanddogs_5340.zip', 'r') as zip_ref: # 'kagglecatsanddogs_5340.zip'라는 이름의 압축 파일을 읽기 모드로 열고 zip_ref 객체로 참조
            zip_ref.extractall('data') # zip_ref 객체를 사용하여 압축 파일 내의 모든 내용을 'data' 디렉토리에 압축 해제

    check_image_dir('data/PetImages/Cat/*.jpg') # 'data/PetImages/Cat' 폴더 내의 모든 '.jpg' 파일을 검사하여 손상된 이미지가 있는지 확인하고, 손상된 이미지는 삭제
    check_image_dir('data/PetImages/Dog/*.jpg') # 'data/PetImages/Dog' 폴더 내의 모든 '.jpg' 파일도 동일하게 검사

    dataset = torchvision.datasets.ImageFolder('data/PetImages',transform=common_transform()) # ImageFolder 클래스를 사용하여 'data/PetImages' 디렉토리의 이미지들을 로드하고 common_transform() 함수를 호출하여 이미지에 적용할 변환을 설정
    trainset, testset = torch.utils.data.random_split(dataset,[20000,len(dataset)-20000]) # 데이터셋을 무작위로 20,000개의 학습 셋과 나머지를 테스트 셋으로 분할
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32) # 학습 데이터셋에 대한 데이터 로더를 생성하고, 배치 크기를 32로 설정
    testloader = torch.utils.data.DataLoader(testset,batch_size=32) # 테스트 데이터셋에 대한 데이터 로더를 생성하고, 배치 크기를 32로 설정
    return dataset, trainloader, testloader # 완성된 데이터셋과 데이터 로더들을 반환





