import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from src.database.transform import *

class RandomFigures(BaseTransform):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.5, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.figures = (cv2.line, cv2.rectangle, cv2.circle)
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

    def apply_image(self, img):
        s = {'state': None}
        if random.uniform(0, 1) > self.p:
            return img, s
        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])
        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img), s

   