import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):

    def __init__(self, root, transforms=None, mode='train'):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.files = os.listdir(os.path.join(self.root, mode))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.mode, self.files[index]))
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
        
        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)
        return img_A, img_B

if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageDataset('data/city_512', train_transforms, mode='test')
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)
    for _, data in enumerate(train_loader):
        print(data[0].shape, data[1].shape)
        