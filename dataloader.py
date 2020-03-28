from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torchvision 

train_root=r'/media/HDD_3TB2/rupali/Dataset/CIFAR10With4Class/Small_Train'
test_root=r'/media/HDD_3TB2/rupali/Dataset/CIFAR10With4Class/Small_Test'
val_root=r'/media/HDD_3TB2/rupali/Dataset/CIFAR10With4Class/Small_Val'



          
        
class clothing_dataloader():  
    def __init__(self, batch_size, num_workers, shuffle):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
   
    def run(self):
        self.transform_train = transforms.Compose([
                #transforms.Resize(256),
                #transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) # meanstd transformation

        self.transform_test = transforms.Compose([
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])    
                
        train_dataset = torchvision.datasets.ImageFolder(train_root, transform=self.transform_train)
        test_dataset = torchvision.datasets.ImageFolder(test_root, transform=self.transform_test)
        val_dataset = torchvision.datasets.ImageFolder(val_root, transform=self.transform_test)
        
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers)              
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)        
        val_loader = DataLoader(
            dataset=val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)            
        return train_loader, val_loader, test_loader