'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse
import sys
home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(home)

from models.vgg_me import VGG_Me
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--l1_lambda', default=1.0, type=float, help='l1 regularization')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/michael/datasets/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/michael/datasets/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
model_name = 'VGG_Me_L1'
net = VGG_Me()

writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), 'runs'), comment=model_name)
dummy_input = torch.rand(13, 3, 32, 32)
writer.add_graph(net, (dummy_input,))

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # add l1 regularization
        for name, param in net.named_parameters():
            if 'classifier.0.weight' in name:
                L1_1 = torch.tensor(param, requires_grad=True)
                L1_2 = torch.norm(L1_1, 1)
                L1_3 = args.l1_lambda * L1_2
                loss = loss + L1_3

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss/(batch_idx+1), correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join(home, 'checkpoint')):
            os.mkdir(os.path.join(home, 'checkpoint'))
        torch.save(state, os.path.join(home, 'checkpoint', model_name.lower() + '.tar'))
        best_acc = acc
    
    return test_loss/(batch_idx+1), correct/total

def analyze():
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.join(home, 'checkpoint')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(home, 'checkpoint', model_name.lower() + '.tar'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('Model accurancy: {}'.format(best_acc))

    Ws = net.classifier.state_dict().items()
    sparsitys = compute_sparsitys(Ws)

    writer.add_text(model_name, 'Total params: {}'.format(compute_total_param_number(net)))
    writer.add_text(model_name, 'Classifier params: {}'.format(compute_param_number(net)))
    writer.add_text(model_name, 'L1 lambda: {}'.format(1.0))
    writer.add_text(model_name, 'Sparsitys: {}'.format(', '.join('{:0.2e}'.format(i) for i in sparsitys)), global_step=start_epoch+200)


for epoch in range(start_epoch, start_epoch+200):

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    
    writer.add_scalars('loss/{}'.format(model_name), {'train_loss': train_loss,
                                                      'test_loss': test_loss}, epoch)
    writer.add_scalars('accurancy/{}'.format(model_name), {'train_acc': train_acc,
                                                            'test_acc': test_acc}, epoch)

analyze()
writer.add_text(model_name, 'The best accurancy on testset is {}'.format(best_acc), global_step=start_epoch+200)