
import numpy as np
from PIL import Image

import torch
from torchvision.transforms.functional import InterpolationMode, resize

class ClassMapToMultiLayers(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, img):
        print(img)
        return img


class ImageResize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
    
    def forward(self, img):
        img_pil = Image.fromarray(np.uint8(img))
        res = resize(img_pil, self.size, self.interpolation, self.max_size, self.antialias)
        return np.asarray(res)

class ImageResizeByCoefficient(torch.nn.Module):
    def __init__(self, coefficient, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.coefficient = coefficient
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias

    def forward(self, img):
        img_size = list(img.shape)
        img_size[0] = (img_size[0] // self.coefficient) * self.coefficient
        img_size[1] = (img_size[1] // self.coefficient) * self.coefficient

        img_pil = Image.fromarray(np.uint8(img))
        res = resize(img_pil, img_size[:2], self.interpolation, self.max_size, self.antialias)
        return np.moveaxis(np.asarray(res), -1, 0)
