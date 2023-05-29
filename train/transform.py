import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# import type
# from typing import *
from typing import Any



class TransMDLTransform(object):
    def __init__(
        self,
        *, 
        train_or_eval: bool = True,
    ) -> None:
        """
        intro:
            transform.
            self.transforms is a callback function.
        
        args:
            :param bool train_or_eval: train's Transform or eval's.
        """
        self.transforms = None

        if train_or_eval:
            list_transform = [ToTensor()]

        else:
            list_transform = [ToTensor()]
            
        
        self.transforms = Compose(
                list_transform,
        )
    
    def __call__(
        self, 
        signal,
        target,
        *args: Any, 
        **kwds: Any
    ) -> Any:
        modify_signal, modify_target = self.transforms(signal, target)
        return (modify_signal, modify_target)



class RandomResize(object):
    """将565大小的image随机放大缩小到[0.5*565, 1.2*565]"""
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size) # 在min和max间随机生成一个数值
        # 将image(PIL.Image类型)缩放到size大小
        image = F.resize(image, size)
        # 选用合适的插值法
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    """将图像随机水平翻转"""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    """竖直方向上翻转"""
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """如果随机数小于0.5则翻转"""
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


def pad_if_smaller(img, size, fill=0):
    """如果img小于size, 则用fill值padding"""
    min_size = min(img.size)    # 取长宽的短边
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img  = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class RandomCrop(object):
    """随机裁剪, 但是裁剪后大小都得为480"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 填充大小到480*480
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill = 255)
        # --- Question
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target
    

class ToTensor(object):
    """将ndarray转为Tensor"""
    def __call__(self, image, target, torch_type=torch.float32):
        # translate `ndarray` to `torch.Tensor`
        image = torch.tensor(image, dtype=torch_type)
        target = torch.tensor(target, dtype=torch_type)
        # image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    """正则化"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    """写了一下Compose的本质, 就是挨个遍历list中的transform, 将其作用在image和target上, 调用__call__()"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
        
if __name__=="__main__":
    X = torch.randn((4, 3, 224, 224))
    Y = torch.zeros((4, 3, 224, 224))

    transMDL_trans = TransMDLTransform(train_or_eval=True)
    a, b = transMDL_trans(X, Y)