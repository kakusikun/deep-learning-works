from src.database.data_format import *

class build_cifar_dataset(Dataset):
    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.data['handle'][index]

        if self.transform is not None:
            img = self.transform(img)
        
        return {'inp': img, 'target': label}
    
    def __len__(self):
        return self.data['n_samples']