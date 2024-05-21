import numpy as np
import os
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from torch import FloatTensor
from torch.utils.data import TensorDataset, DataLoader
from VanillaCNN import BaseCNN
from util import prepared_train,prepared_test

def train(dataloader,model,criterion,optimizer):
    # 모델 훈련
    num_epochs = 5
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for batch_data, batch_target in dataloader:
            # Forward pass
            outputs = model(batch_data)

            loss = criterion(outputs.squeeze(), batch_target.float())
            
            # Backward pass and optimization
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
    return model,optimizer


def training(data:FloatTensor, target:FloatTensor):
    # TensorDataset 생성
    dataset = TensorDataset(data, target)
    # DataLoader 생성
    batch_size = 64
    training = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return training


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # 파일 경로와 오디오 파라미터 설정
    train_x,train_y = prepared_train(os.getcwd()+"\\data\\raw16k\\train\\")
    data = FloatTensor(train_x).to(device)
    target = FloatTensor(train_y).to(device)
    train_loader = training(data,target)

    test_x,test_y = prepared_test(os.getcwd()+"\\data\\raw16k\\test\\","./fmc_test_ref.csv")

    # 모델 인스턴스 생성
    model = BaseCNN().to(device)
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model,optimizer = train(train_loader,model,criterion,optimizer)
    
    pred = model(test_y)








    
