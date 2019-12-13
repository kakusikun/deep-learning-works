from database.datasets import *
from PIL import Image

class build_image_dataset(data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        img_path, label = self.dataset[index]

        img = Image.open(img_path)        
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.dataset)