import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import Image
import numpy as np
import math

_MAX_LEVEL = 10

# affine transforms
def apply_A(pt, A):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(A, new_pt)
    return new_pt[:2]

def identity_A(level, **kwargs):
    return np.array([1, 0, 0, 0, 1, 0]).reshape(2, 3).astype(np.float)

def shear_x_A(level, **kwargs):
    return np.array([1, -level, 0, 0, 1, 0]).reshape(2, 3).astype(np.float)

def shear_y_A(level, **kwargs):
    return np.array([1, 0, 0, -level, 1, 0]).reshape(2, 3).astype(np.float)

def translate_x_A(level, **kwargs):
    return np.array([1, 0, -level, 0, 1, 0]).reshape(2, 3).astype(np.float)

def translate_y_A(level, **kwargs):
    return np.array([1, 0, 0, 0, 1, -level]).reshape(2, 3).astype(np.float)

def rotate_A(level, shape, **kwargs):
    # copy from https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Image.rotate
    angle = level % 360.0

    w, h = shape

    post_trans = (0, 0)
    rotn_center = (w / 2.0, h / 2.0)
    angle = math.radians(angle)
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(
        -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
    )
    matrix[2] += rotn_center[0]
    matrix[5] += rotn_center[1]

    return np.array(matrix).reshape(2, 3).astype(np.float)

# img transforms
def shear_x_op(img, level):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, level, 0, 0, 1, 0))

def shear_y_op(img, level):  # [-0.3, 0.3]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, level, 1, 0))

def translate_x_op(img, level):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, level, 0, 1, 0))

def translate_y_op(img, level):  # [-150, 150] => percentage: [-0.45, 0.45]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, level))

def rotate_op(img, level):  # [-30, 30]
    return img.rotate(level)

def autocontrast_op(img, _):
    return PIL.ImageOps.autocontrast(img)

def invert_op(img, _):
    return PIL.ImageOps.invert(img)

def equalize_op(img, _):
    return PIL.ImageOps.equalize(img)

def solarize_op(img, level):  # [0, 256]
    return PIL.ImageOps.solarize(img, level)

def solarize_add_op(img, level=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + level
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

def posterize_op(img, level):  # [4, 8]
    return PIL.ImageOps.posterize(img, level)

def contrast_op(img, level):  # [0.1,1.9]
    return PIL.ImageEnhance.Contrast(img).enhance(level)   

def color_op(img, level):  # [0.1,1.9]
    return PIL.ImageEnhance.Color(img).enhance(level)

def brightness_op(img, level):  # [0.1,1.9]
    return PIL.ImageEnhance.Brightness(img).enhance(level)

def sharpness_op(img, level):  # [0.1,1.9]
    return PIL.ImageEnhance.Sharpness(img).enhance(level)

def cutout_op(img, level):  # [0, 40] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if level < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - level / 2.))
    y0 = int(max(0, y0 - level / 2.))
    x1 = min(w, x0 + level)
    y1 = min(h, y0 + level)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def identity_op(img, level):
    return img

# level of transforms
def _random_negate_level(level):
    if random.random() > 0.5:
        level = -level
    return level

def no_level(level, **kwargs):
    return None

def shear_level(level, value=0.3, **kwargs):
    level = (level/_MAX_LEVEL) * value
    # Flip level to negative with 50% chance.
    level = _random_negate_level(level)
    return level

def translate_level(level, value=3, **kwargs):
    translate_const = kwargs['size'][0] // value
    level = (level/_MAX_LEVEL) * float(translate_const)
    level = _random_negate_level(level)
    return level

def enhance_level(level, **kwargs):
    return (level/_MAX_LEVEL) * 1.8 + 0.1

def posterize_level(level, **kwargs):
    return int((level/_MAX_LEVEL) * 4)

def solarize_level(level, **kwargs):
    return int((level/_MAX_LEVEL) * 256)

def solarized_add_level(level, **kwargs):
    return int((level/_MAX_LEVEL) * 110)

def rotate_level(level, value=30, **kwargs):
    level = (level/_MAX_LEVEL) * value
    level = _random_negate_level(level)
    return level

def cutout_level(level, **kwargs):
    cutout_const = kwargs['size'][0]
    return int((level/_MAX_LEVEL) * cutout_const)

AUG_LEVELS = {
    'AutoContrast': no_level,
    'Equalize': no_level,
    'Invert': no_level,
    'Rotate': rotate_level,
    'Posterize': posterize_level,
    'Solarize': solarize_level,
    'SolarizeAdd': solarized_add_level,
    'Color': enhance_level,
    'Contrast': enhance_level,
    'Brightness': enhance_level,
    'Sharpness': enhance_level,
    'ShearX': shear_level,
    'ShearY': shear_level,
    'Cutout': cutout_level,
    'TranslateX': translate_level,
    'TranslateY': translate_level
}

AUG_OPS = {
    'AutoContrast': autocontrast_op,
    'Equalize': equalize_op,
    'Invert': invert_op,
    'Rotate': rotate_op,
    'Posterize': posterize_op,
    'Solarize': solarize_op,
    'SolarizeAdd': solarize_add_op,
    'Color': color_op,
    'Contrast': contrast_op,
    'Brightness': brightness_op,
    'Sharpness': sharpness_op,
    'ShearX': shear_x_op,
    'ShearY': shear_y_op,
    'Cutout': cutout_op,
    'TranslateX': translate_x_op,
    'TranslateY': translate_y_op
}

AUG_AS = {
    'AutoContrast': identity_A,
    'Equalize': identity_A,
    'Invert': identity_A,
    'Rotate': rotate_A,
    'Posterize': identity_A,
    'Solarize': identity_A,
    'SolarizeAdd': identity_A,
    'Color': identity_A,
    'Contrast': identity_A,
    'Brightness': identity_A,
    'Sharpness': identity_A,
    'ShearX': shear_x_A,
    'ShearY': shear_y_A,
    'Cutout': identity_A,
    'TranslateX': translate_x_A,
    'TranslateY': translate_y_A
}

RANDAUG_OPS_NAME = [
    'AutoContrast', 'Equalize', 'Invert', 'Rotate',
    'Posterize', 'Solarize', 'SolarizeAdd', 'Color',
    'Contrast', 'Brightness', 'Sharpness', 'ShearX',
    'ShearY', 'Cutout', 'TranslateX', 'TranslateY'
]

AUGMIX_OPS_NAME = [
    'AutoContrast', 'Equalize', 'Rotate',
    'Posterize', 'Solarize', 'ShearX',
    'ShearY', 'TranslateX', 'TranslateY'
]