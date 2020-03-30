import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class RandomGrid(BaseTransform):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def apply_image(self, img):
        s = {'state': None}
        if random.uniform(0, 1) > self.p:
            return img, s
        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color
        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)
        return self.draw_grid(img, grid_size, color, thickness, angle), s

    @staticmethod
    def draw_grid(image, grid_size, color, thickness, angle):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

   