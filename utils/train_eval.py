import os
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init


# Inicialize paramets for net
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias.data, 0)

# Training
def train(net, criterion, optimizer, loader, epoch:int=1, device:str = "cuda"):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(loader, unit="batch") as pbar_loader:
        for batch_idx, (inputs, targets) in enumerate(pbar_loader):
            pbar_loader.set_description(f"Training model for epoch {epoch}")
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar_loader.set_postfix(loss=train_loss/(batch_idx+1), accuracy=100.*correct/total)

# Evaluate
def test(net, criterion, optimizer, loader, epoch:int=1, device: str = "cuda"):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(loader, unit="batch") as pbar_loader:
            for batch_idx, (inputs, targets) in enumerate(pbar_loader):
                pbar_loader.set_description(f"Testing model for epoch {epoch}")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                # progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                pbar_loader.set_postfix(loss=test_loss/(batch_idx+1), accuracy=100.*correct/total)
    acc = 100.*correct/total
    return acc

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
