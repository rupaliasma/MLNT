from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torch
import torchvision.transforms as transforms
from customtransforms import RandomHorizontalFlipTensor, RandomVerticalFlipTensor
import random
import numpy as np
from PIL import Image
import torchvision 
from tqdm import tqdm


def get_all_data(dataset, num_workers=30, shuffle=False):
    dataset_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size=dataset_size,
                             num_workers=num_workers, shuffle=shuffle)
    all_data = {}
    for i_batch, sample_batched in tqdm(enumerate(data_loader)):
        all_data = sample_batched
    return all_data

def flip_label(y, pattern, ratio, one_hot=True):
    #Origin: https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/data.py
    #y: true label, one hot
    #pattern: 'pair' or 'sym'
    #p: float, noisy ratio
    
    #convert one hot label to int
    if one_hot:
        y = np.argmax(y,axis=1)#[np.where(r==1)[0][0] for r in y]
    n_class = max(y)+1
    
    #filp label
    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            y[i] = np.random.choice([y[i],(y[i]+1)%n_class],p=[1-ratio,ratio])            
            
    #convert back to one hot
    if one_hot:
        y = np.eye(n_class)[y]
    return y

def get_class_subset(imgs, lbls, n_classes):
    selectedIdx = lbls < n_classes
    return imgs[selectedIdx], lbls[selectedIdx]


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

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, target_transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
          
        
class DataLoadersCreator():  
    def __init__(self, batch_size, num_workers, shuffle, cifar_root=r'/media/HDD2TB/rupali/Work104/Dataset/CIFAR10', 
    noise_pattern='sym', noise_ratio=0.5, n_classes=None):
    # def __init__(self, batch_size, num_workers, shuffle, cifar_root=r'F:\CIFAR10' /media/HDD_3TB2/rupali/Dataset/CIFAR10, noise_pattern='sym', noise_ratio=0.5):
    #set noise_pattern to None if no noise is intended to be added in the trainset
    #set n_classes to None if you want all the classes

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.cifar_root = cifar_root
        self.noise_pattern = noise_pattern
        self.noise_ratio = noise_ratio
        self.n_classes = n_classes
   
    def run(self):
        self.transform_augments = transforms.Compose([
                #transforms.RandomSizedCrop(224),
                RandomHorizontalFlipTensor(),
            ]) # meanstd transformation

        self.transform_noaugment = transforms.Compose([
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])    

        trainval_dataset = torchvision.datasets.CIFAR10(self.cifar_root, train=True, transform=self.transform_noaugment, target_transform=None, download=True)
        test_dataset = torchvision.datasets.CIFAR10(self.cifar_root, train=False, transform=self.transform_noaugment, target_transform=None, download=True)
                

        lengths = [int(len(trainval_dataset)*0.8), int(len(trainval_dataset)*0.2)]
        train_dataset, val_dataset = random_split(trainval_dataset, lengths)
        
        train_imgs, train_targets = get_all_data(train_dataset, num_workers=self.num_workers)

        if self.n_classes is not None:
            train_imgs, train_targets = get_class_subset(train_imgs, train_targets, self.n_classes)
        
        if self.noise_pattern is not None:
            original_train_targets_np = train_targets.numpy()
            train_targets_np = flip_label(original_train_targets_np.copy(), pattern=self.noise_pattern, ratio=self.noise_ratio, one_hot=False)
            n_noisy_labels = original_train_targets_np.size - (original_train_targets_np==train_targets_np).sum()
            print('no of noisy labels : '+str(n_noisy_labels))
            train_targets = torch.from_numpy(train_targets_np)

        train_dataset = CustomTensorDataset([train_imgs, train_targets], transform=self.transform_augments)

        val_imgs, val_targets = get_all_data(val_dataset, num_workers=self.num_workers)
        if self.n_classes is not None:
            val_imgs, val_targets = get_class_subset(val_imgs, val_targets, self.n_classes)
        val_dataset = TensorDataset(val_imgs, val_targets)

        if self.n_classes is not None:
            test_imgs, test_targets = get_all_data(test_dataset, num_workers=self.num_workers)
            test_imgs, test_targets = get_class_subset(test_imgs, test_targets, self.n_classes)
            test_dataset = TensorDataset(test_imgs, test_targets)
        
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


if __name__ == "__main__":
    x=DataLoadersCreator(batch_size=10, num_workers=0, shuffle=True)
    tr, te, v = x.run()
    for i_batch, sample_batched in tqdm(enumerate(tr)):
        print('d')
    print('shit')