from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torchvision 



class CustomCIFAR(Dataset):
    def __init__(self, subset, transform=None, target_transform=None):
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        img, target = self.subset[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        
    def __len__(self):
        return len(self.subset)

          
        
class clothing_dataloader():  
    def __init__(self, batch_size, num_workers, shuffle, cifar_root=r'/media/HDD_3TB2/rupali/Dataset/CIFAR10'):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.cifar_root = cifar_root
   
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

        trainval_dataset = torchvision.datasets.CIFAR10(self.cifar_root, train=True, transform=None, target_transform=None, download=True)
        test_dataset = torchvision.datasets.CIFAR10(self.cifar_root, train=False, transform=self.transform_test, target_transform=None, download=True)
                

        lengths = [int(len(trainval_dataset)*0.8), int(len(trainval_dataset)*0.2)]
        subsetA, subsetB = random_split(trainval_dataset, lengths)
        train_dataset = CustomCIFAR(
            subsetA, transform=self.transform_train)
        )
        val_dataset = CustomCIFAR(
            subsetB, transform=self.transform_test)
        )
        
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