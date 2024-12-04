import numpy as np


def accuracy(pred, target):
    error_pixel = pred - target
    bs, channal, h, w = pred.shape
    num_pixel = bs * channal *h * w
    error_pixel_th = np.abs(error_pixel) <= 0.05
    num_right = np.sum(error_pixel_th)
    acc = num_right/ num_pixel
    return acc
