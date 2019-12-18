from database.datasets import *
from PIL import Image

class build_image_dataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        img_path, label = self.data[index]

        img = Image.open(img_path)        
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return {'inp': img, 'target': label}
    
    def __len__(self):
        return len(self.data)