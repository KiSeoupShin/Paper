from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import VGG16
from pathlib import Path

import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def train_model(model, batch_size, data_root, learning_rate, epochs, device, dir_checkpoint):
    model = model.to(device)
    
    # image 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # dataset 구성
    train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # dataloader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # optimizer & loss function 설정
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    # training start
    for epoch in range(epochs + 1):
        model.train()
        epoch_loss = 0
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(train_dataloader)
            del x, y
        
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for x, y in valid_dataloader:
                x = x.to(device)
                y = y.to(device)

                y_pred = model.forward(x)
                loss = criterion(y_pred, y)

                test_loss += loss.item() / len(valid_dataloader)
                del x, y

        print(f'epoch : {epoch}, loss : {test_loss}')

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, str(dir_checkpoint + 'checkpoint' + '.pth'))

if __name__ == '__main__':
    model = VGG16(n_classes=config.CLASS)
    train_model(model=model, batch_size=config.BATCH_SIZE, data_root=config.DATA_ROOT,
                learning_rate=config.LEARNING_RATE, epochs=config.EPOCH, device=config.DEVICE,
                dir_checkpoint='./checkpoint/')