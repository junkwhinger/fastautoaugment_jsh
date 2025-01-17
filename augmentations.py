# References for the augmentation classes
# https://github.com/tensorflow/models/blob/master/research/autoaugment/augmentation_transforms.py
# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py

import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import torchvision.transforms as TF

# flips lambda(v)
random_mirror = True

class ShearX(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):
        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])
        if toss:
            _min = -0.3
            _max = 0.3
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            if random_mirror and random.random() > 0.5:
                mag = -mag
            return img.transform(img.size, Image.AFFINE, (1, mag, 0, 0, 1, 0))
        else:
            return img


class ShearY(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):
        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])
        if toss:
            _min = -0.3
            _max = 0.3
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            if random_mirror and random.random() > 0.5:
                mag = -mag
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, mag, 1, 0))
        else:
            return img


class TranslateX(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = -0.45
            _max = 0.45
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            if random_mirror and random.random() > 0.5:
                mag = -mag
            mag = mag * img.size[0]
            return img.transform(img.size, Image.AFFINE, (1, 0, mag, 0, 1, 0))
        else:
            return img


class TranslateY(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):
        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = -0.45
            _max = 0.45
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            if random_mirror and random.random() > 0.5:
                mag = -mag
            mag = mag * img.size[1]
            return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, mag))
        else:
            return img


class Rotate(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = -30
            _max = 30
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return img.rotate(mag)
        else:
            return img


class AutoContrast(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            return ImageOps.autocontrast(img)
        else:
            return img


class Invert(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            return ImageOps.invert(img)
        else:
            return img


class Equalize(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            return ImageOps.equalize(img)
        else:
            return img


class Solarize(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0
            _max = 256
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            assert 0 <= mag <= 256
            return ImageOps.solarize(img, mag)
        else:
            return img


class Posterize(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 4
            _max = 8
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return ImageOps.posterize(img, int(mag))
        else:
            return img


class Contrast(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0.1
            _max = 1.9
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return ImageEnhance.Contrast(img).enhance(mag)
        else:
            return img


class Color(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0.1
            _max = 1.9
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return ImageEnhance.Color(img).enhance(mag)
        else:
            return img


class Brightness(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0.1
            _max = 1.9
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return ImageEnhance.Brightness(img).enhance(mag)
        else:
            return img


class Sharpness(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0.1
            _max = 1.9
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            return ImageEnhance.Sharpness(img).enhance(mag)
        else:
            return img


class Cutout(object):
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def __call__(self, img):

        toss = np.random.choice([1, 0], p=[self.p, 1 - self.p])

        if toss:
            _min = 0.0
            _max = 0.2
            mag = self.v * (_max - _min) + _min
            assert _min <= mag <= _max
            if mag <= 0.0:
                return img

            mag *= img.size[0]

            return CutoutAbs(img, mag)
        else:
            return img


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, color)
    return img


class CustomCompose(TF.Compose):
    """
    CustomCompose Class for Bayesian Optimization
    """
    def __init__(self, base):
        super(CustomCompose, self).__init__(base)
        self.base = base
        self.transforms = base.copy()

    def reset(self):
        """
        Method that resets self.transforms
        so that it can take new extra augmentation options
        :return:
        """
        self.transforms = None
        self.transforms = self.base

    def build(self, sub_policy):
        """
        Method that parse the input sub_policy
        and insert them into the base transform function
        :param sub_policy: dictionary of a sub_policy
        """
        print(sub_policy)
        for op_dict in sub_policy['sub_policy']:
            op_key = list(op_dict.keys())[0]
            if op_dict[op_key][op_key + "_v"] is not None:
                aug_fn = (eval(op_key)(op_dict[op_key][op_key + "_p"], op_dict[op_key][op_key + "_v"]))
            else:
                aug_fn = (eval(op_key)(op_dict[op_key][op_key + "_p"], 0))

            self.transforms.insert(0, aug_fn)


class FAAaugmentation(object):
    """
    FAAaugmentation Class that randomly selects a policy
    and transforms a given image when it's called
    """
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        img_copied = img.copy()
        policy_idx_list = (list(self.policies.keys()))
        policy_idx = random.choice(policy_idx_list)
        chosen_policy = self.policies[policy_idx]

        for op_name, op_vals in chosen_policy.items():
            aug_fn = (eval(op_name)(op_vals[0], op_vals[1]))
            img_copied = aug_fn(img_copied)

        return img_copied
