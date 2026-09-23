# Augmix augmentation dung cho MEMO (Zhang et al., NeurIPS 2022) -- sao chep tu
# repo chinh thuc cua tac gia (memo/cifar-10-exps/utils/third_party.py va
# memo/imagenet-exps/utils/third_party.py, hai file goc chi khac nhau hang so
# kich thuoc anh 32 vs 224 hardcode). O day gop lai thanh 1 ham factory tham so
# hoa theo img_size de dung chung cho CIFAR-10/100-C (32) va Tiny-ImageNet-C/PACS (224).
# Nguon augmix: https://github.com/google-research/augmix
import numpy as np
import torch
from PIL import ImageOps, Image
from torchvision import transforms


def _int_parameter(level, maxval):
    return int(level * maxval / 10)


def _float_parameter(level, maxval):
    return float(level) * maxval / 10.


def _rand_lvl(n):
    return np.random.uniform(low=0.1, high=n)


def _make_ops(img_size):
    def autocontrast(pil_img, level=None):
        return ImageOps.autocontrast(pil_img)

    def equalize(pil_img, level=None):
        return ImageOps.equalize(pil_img)

    def rotate(pil_img, level):
        degrees = _int_parameter(_rand_lvl(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)

    def solarize(pil_img, level):
        level = _int_parameter(_rand_lvl(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(pil_img, level):
        level = _float_parameter(_rand_lvl(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((img_size, img_size), Image.AFFINE, (1, level, 0, 0, 1, 0),
                                  resample=Image.BILINEAR, fillcolor=128)

    def shear_y(pil_img, level):
        level = _float_parameter(_rand_lvl(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, 0, level, 1, 0),
                                  resample=Image.BILINEAR, fillcolor=128)

    def translate_x(pil_img, level):
        level = _int_parameter(_rand_lvl(level), img_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, level, 0, 1, 0),
                                  resample=Image.BILINEAR, fillcolor=128)

    def translate_y(pil_img, level):
        level = _int_parameter(_rand_lvl(level), img_size / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((img_size, img_size), Image.AFFINE, (1, 0, 0, 0, 1, level),
                                  resample=Image.BILINEAR, fillcolor=128)

    def posterize(pil_img, level):
        level = _int_parameter(_rand_lvl(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    return [
        autocontrast, equalize,
        lambda x: rotate(x, 1), lambda x: solarize(x, 1),
        lambda x: shear_x(x, 1), lambda x: shear_y(x, 1),
        lambda x: translate_x(x, 1), lambda x: translate_y(x, 1),
        lambda x: posterize(x, 1),
    ]


def make_augmix(img_size, mean, std):
    """Tra ve ham aug(pil_image) -> tensor da augmix, dung cho adapt_single cua MEMO."""
    augmentations = _make_ops(img_size)
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if img_size <= 32:
        preaugment = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        preaugment = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])

    def aug(x_orig):
        x_orig = preaugment(x_orig)
        x_processed = preprocess(x_orig)
        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_processed)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(augmentations)(x_aug)
            mix += w[i] * preprocess(x_aug)
        mix = m * x_processed + (1 - m) * mix
        return mix

    return aug
