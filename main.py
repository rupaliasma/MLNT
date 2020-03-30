import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models as models
import math
import os
import sys
import time
import argparse
import datetime
from torch.autograd import Variable
from collections import OrderedDict
import dataloader
import random

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import scipy.io as sio
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.0008, type=float, help='learning_rate')
parser.add_argument('--meta_lr', default=0.02, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=3, type=int, help='number of random perturbations')
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--start_iter', default=500, type=int)
parser.add_argument('--mid_iter', default=2000, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--eps', default=0.99, type=float, help='Running average of model weights')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='MLNT')
parser.add_argument('--nclass', default=10, type=int)
parser.add_argument('--checkpoint', default='cross_entropy')
parser.add_argument('--drop_prob', default=0.0, type=float)
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

writer = SummaryWriter(log_dir='TBLogsMain')

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
# Training
def train(epoch):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation        
        
        class_loss = criterion(outputs, targets)  # Loss
        class_loss.backward(retain_graph=True)  

        if batch_idx>args.start_iter or epoch>1:
            if batch_idx>args.mid_iter or epoch>1:
                args.eps=0.999
                alpha = args.alpha
            else:
                u = (batch_idx-args.start_iter)/(args.mid_iter-args.start_iter)
                alpha = args.alpha*math.exp(-5*(1-u)**2)          
          
            if init:
                init = False
                for param,param_tch in zip(net.parameters(),tch_net.parameters()): 
                    param_tch.data.copy_(param.data)                    
            else:
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(args.eps).add_((1-args.eps), param.data)   
            
            _,feats = pretrain_net(inputs,get_feat=True)
            tch_outputs = tch_net(inputs,get_feat=False)
            p_tch = F.softmax(tch_outputs,dim=1)
            p_tch.detach_()
            
            for i in range(args.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                for n in range(int(targets.size(0)*args.perturb_ratio)):
                    num_neighbor = args.nclass - 1
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1,feat.size(0))
                    feat.data = feat.data.expand(targets.size(0),feat.size(0))
                    dist = torch.sum((feat-feats)**2,dim=1)
                    _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1,num_neighbor)]]
                    
                fast_loss = criterion(outputs,targets_fast)

                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                for grad in grads:
                    grad.detach()
                    #grad.requires_grad = False  
                    grad.detach().requires_grad_(False)
   
                fast_weights = OrderedDict((name, param - args.meta_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                
                fast_out = net.forward(inputs,fast_weights)  
    
                logp_fast = F.log_softmax(fast_out,dim=1)                
                consistent_loss = consistent_criterion(logp_fast,p_tch)
                consistent_loss = consistent_loss*alpha/args.num_fast 
                consistent_loss.backward(retain_graph=True)
                
        optimizer.step() # Optimizer update

        train_loss += class_loss.data[0]      
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.data[0], 100.*correct/total))
        sys.stdout.flush()
        writer.add_scalar('Loss/train', class_loss.data[0], epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Accuracy/train', 100.*correct/total, epoch*len(train_loader)+batch_idx)
        if batch_idx%1000==0:
            val(epoch,batch_idx)
            val_tch(epoch,batch_idx)
            net.train()
            tch_net.train()            
            
            
def val(epoch,iteration):
    global best
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()        

        writer.add_scalar('Loss/val', loss.data[0], epoch*len(val_loader)+batch_idx)
        writer.add_scalar('Accuracy/val', 100.*correct/total, epoch*len(val_loader)+batch_idx)

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, iteration, loss.data[0], acc))
    record.write('Epoch #%d Batch #%3d  Acc: %.2f' %(epoch,iteration,acc))

    #Saving checkpoint always
    print('| Saving Current Model (net)...')
    save_point = './checkpoint_current/%s.net.pth.tar'%(args.id)
    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimzer': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc,
        'best_acc': best
    }, save_point)

    #Saving current model
    if acc > best:
        best = acc
        print('| Saving Best Model (net)...')
        save_point = './checkpoint/%s.main.pth.tar'%(args.id)
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimzer': optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best,
        }, save_point)       

def val_tch(epoch,iteration):
    global best
    tch_net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = tch_net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, iteration, loss.data[0], acc))
    record.write(' | tchAcc: %.2f\n' %acc)
    record.flush()

    #Saving regular checkpoint
    print('| Saving Current Model (tchnet)...')
    save_point = './checkpoint_current/%s.tchnet.pth.tar'%(args.id)
    save_checkpoint({
        'state_dict': tch_net.state_dict(),
        'acc': acc,
        'best_acc': best,
    }, save_point) 

    #Saving best model
    if acc > best:
        best = acc
        print('| Saving Best Model (tchnet)...')
        save_point = './checkpoint/%s.main.pth.tar'%(args.id)
        save_checkpoint({
            'state_dict': tch_net.state_dict(),
            'best_acc': best,
        }, save_point)        

def saveIMGs(path, ip, op, pred, target):
    for i in range(ip.shape[0]):
        ip1 = ip[i]
        op1 = op[i]
        pred1 = pred[i]
        target1 = target[i]
        path1 = path+'_'+str(i)+'_'+str(pred1)+'_'+str(target1)+'.png'
        ip1 = np.transpose(ip1, [1,2,0])
        ip1 = (ip1-ip1.min()) / (ip1.max()-ip1.min())
        ip1 *= 255

        cv2.imwrite(path1, ip1) 



def test(save_path):
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

        ip = inputs.detach().cpu().numpy()
        op = outputs.detach().cpu().numpy()

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        writer.add_scalar('Loss/test', loss.data[0], epoch*len(test_loader)+batch_idx)
        writer.add_scalar('Accuracy/test', 100.*correct/total, epoch*len(test_loader)+batch_idx)

        pred = predicted.detach().cpu().numpy()

        targ = targets.detach().cpu().numpy()
        
        mat = {'ip':ip, 'op':op, 'pred':pred, 'targ':targ}
        sio.savemat(os.path.join(save_path, str(batch_idx)+'.mat'), mat)

        saveIMGs(os.path.join(save_path, 'imgs', str(batch_idx)), ip, op, pred, targ)

    test_acc = 100.*correct/total   
    print('* Test results : Acc@1 = %.2f%%' %(test_acc))
    record.write('\nTest Acc: %f\n'%test_acc)
    record.flush()
    
record=open('./checkpoint/'+args.id+'.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.write('batch size: %f\n'%args.batch_size)
record.write('start iter: %d\n'%args.start_iter)  
record.write('mid iter: %d\n'%args.mid_iter)  
record.flush()

     
loader = dataloader.DataLoadersCreator(batch_size=args.batch_size,num_workers=5,shuffle=True, noise_pattern=args.noise_pattern, noise_ratio=args.noise_ratio)
train_loader,val_loader,test_loader = loader.run()

best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')
net = models.resnet50(pretrained=True, do=args.drop_prob)
net.fc = nn.Linear(2048,args.nclass)
tch_net = models.resnet50(pretrained=True, do=args.drop_prob)
tch_net.fc = nn.Linear(2048,args.nclass)
pretrain_net = models.resnet50(pretrained=True, do=args.drop_prob)
pretrain_net.fc = nn.Linear(2048,args.nclass)
test_net = models.resnet50(pretrained=True, do=args.drop_prob)
test_net.fc = nn.Linear(2048,args.nclass)

print('| load pretrain from checkpoint...')
checkpoint = torch.load('./checkpoint/%s.baseline.pth.tar'%args.checkpoint)
pretrain_net.load_state_dict(checkpoint['state_dict'])

if use_cuda:
    net.cuda()
    tch_net.cuda()
    pretrain_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True
pretrain_net.eval()    

for param in tch_net.parameters(): 
    param.requires_grad = False   
for param in pretrain_net.parameters(): 
    param.requires_grad = False 

criterion = nn.CrossEntropyLoss()
consistent_criterion = nn.KLDivLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

start_epoch = 1

if args.resume:
    checkpoint = torch.load('./checkpoint_current/%s.net.pth.tar'%args.id)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']

    tchcheckpoint = torch.load('./checkpoint_current/%s.tchnet.pth.tar'%args.id)
    tch_net.load_state_dict(tchcheckpoint['state_dict'])

    best_model = torch.load('./checkpoint/%s.main.pth.tar'%args.id)
    best = best_model['best_acc']


print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

for epoch in range(start_epoch, 1+args.num_epochs):
    train(epoch)

print('\nTesting model')
best_model = torch.load('./checkpoint/%s.main.pth.tar'%args.id)
test_net.load_state_dict(best_model['state_dict'])
test(save_path=r'/media/HDD_3TB2/rupali/Code/MLNT-master/checkpoint/results')

record.close()