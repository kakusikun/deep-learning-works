from src.database.data_format import *
from PIL import Image

class build_reid_dataset(Dataset):
    def __init__(self, data, transform=None, return_indice=False, **kwargs):
        self.data = data
        self.transform = transform
        self.return_indice = return_indice
           
    def __getitem__(self, index):
        img_path, pid, camid = self.data['indice'][index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
            if isinstance(img, tuple):
                img = img[0]
        if self.return_indice:
            return img, pid, camid, index
            
        return {'inp': img, 'pid': pid, 'camid': camid}

    def __len__(self):
        return self.data['n_samples']