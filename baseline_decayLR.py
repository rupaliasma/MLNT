
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models as models

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
import argparse
import datetime
import numpy as np
import random

from torch.autograd import Variable

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import dataloader

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--lrdecay_nepoch', default=25, type=float, help='decay learning_rate after every n epoch')
parser.add_argument('--lrdecay', default=0.1, type=float, help='decay rate of learning_rate')
parser.add_argument('--start_epoch', default=1, type=int) #We won't use it, as it will be decided by the checkpoint if resuming
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid',default=1, type=int)
parser.add_argument('--nclass', default=10, type=int)
parser.add_argument('--id', default='do25_sym50_SGD_lrdecay')
parser.add_argument('--drop_prob', default=0.25, type=float)
parser.add_argument('--noise_pattern', default='sym')
parser.add_argument('--noise_ratio', default=0.5, type=float)
parser.add_argument('--resume', default=False, type=bool)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(args.gpuid)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()


writer = SummaryWriter(log_dir='./TBLogs/Baseline_%s/'%(args.id)) 


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by lrDecayRate every lrDecayNEpoch epochs"""

    lr = args.lr * (args.lrdecay ** (epoch // args.lrdecay_nepoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = adjust_learning_rate(optimizer, epoch-1)

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()
        writer.add_scalar('Loss/train', loss.data[0], epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Accuracy/train', 100.*correct/total, epoch*len(train_loader)+batch_idx)
        if batch_idx%1000==0:
            val(epoch)
            net.train()
            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        writer.add_scalar('Loss/val', loss.data[0], epoch*len(val_loader)+batch_idx)
        writer.add_scalar('Accuracy/val', 100.*correct/total, epoch*len(val_loader)+batch_idx)

    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    record.write('Validation Acc: %f\n'%acc)

    #Saving checkpoint always
    print('| Saving Current Model (net)...')
    save_point = './checkpoint_current/%s.baseline.pth.tar'%(args.id)
    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimzer': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'best_acc': best_acc
    }, save_point)

    # Save checkpoint when best model
    record.flush()    
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_point = './checkpoint/%s.baseline.pth.tar'%(args.id)
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimzer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
        }, save_point) 

def test():
    global test_acc
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        writer.add_scalar('Loss/test', loss.data[0], epoch*len(test_loader)+batch_idx)
        writer.add_scalar('Accuracy/test', 100.*correct/total, epoch*len(test_loader)+batch_idx)

    acc = 100.*correct/total   
    test_acc = acc
    record.write('Test Acc: %f\n'%acc)
    
os.makedirs('checkpoint', exist_ok = True)     
os.makedirs('checkpoint_current', exist_ok = True)     
record=open('./checkpoint/baseline_'+args.id+'_test.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.flush()
     
loader = dataloader.DataLoadersCreator(batch_size=args.batch_size,num_workers=5,shuffle=True, noise_pattern=args.noise_pattern, noise_ratio=args.noise_ratio)
train_loader,val_loader,test_loader = loader.run()

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
net = models.resnet50(pretrained=True, do=args.drop_prob)
net.fc = nn.Linear(2048,args.nclass)
test_net = models.resnet50(pretrained=True, do=args.drop_prob)
test_net.fc = nn.Linear(2048,args.nclass)
if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.optim_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
elif args.optim_type == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-3)
else:
    sys.exit('Invalid Optimzer Choise')

start_epoch = 1

if args.resume:
    checkpoint = torch.load('./checkpoint_current/%s.baseline.pth.tar'%args.id)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

for epoch in range(start_epoch, 1+args.num_epochs):
    train(epoch)
    val(epoch)

print('\nTesting model')
checkpoint = torch.load('./checkpoint/%s.baseline.pth.tar'%args.id)
test_net.load_state_dict(checkpoint['state_dict'])
test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
record.write('Test Acc: %.2f\n' %test_acc)
record.flush()
record.close()
