from src.database.data_format import *
from PIL import Image

class build_image_dataset(Dataset):
    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.transform = transform
    
    def __getitem__(self, index):
        img_path, label = self.data['indice'][index]

        img = Image.open(img_path)    
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return {'inp': img, 'target': label}
    
    def __len__(self):
        return self.data['n_samples']
