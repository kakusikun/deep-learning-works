
import torchvision.transforms as T

def build_transform(cfg, isTrain=True):
    bagTransforms = []
    
    if isTrain:
        if cfg.TRANSFORM.RESIZE:
            bagTransforms.append(T.Resize(size=cfg.INPUT.IMAGE_SIZE))

        if cfg.TRANSFORM.HFLIP:
            bagTransforms.append(T.RandomHorizontalFlip(p=cfg.INPUT.PROB))
            
        if cfg.TRANSFORM.RANDOMCROP:
            bagTransforms.append(T.RandomCrop(size=cfg.INPUT.IMAGE_SIZE, padding=cfg.INPUT.IMAGE_PAD))   
           
    else:
        if cfg.TRANSFORM.RESIZE:
            bagTransforms.append(T.Resize(size=cfg.INPUT.IMAGE_SIZE))
        if cfg.TRANSFORM.RANDOMCROP:
            bagTransforms.append(T.RandomCrop(size=cfg.INPUT.IMAGE_SIZE, padding=cfg.INPUT.IMAGE_PAD))   
        
    bagTransforms.append(T.ToTensor())

    if cfg.TRANSFORM.NORMALIZE:
        bagTransforms.append(T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))

            
    transform = T.Compose(bagTransforms)

    return transform
