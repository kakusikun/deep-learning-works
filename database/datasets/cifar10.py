from database.datasets import *

class build_cifar_dataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.data[index]

        if self.transform is not None:
            img = self.transform(img)
        
        return {'inp': img, 'target': label}
    
    def __len__(self):
        return len(self.data)