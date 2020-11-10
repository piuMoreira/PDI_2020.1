import numpy as np
import util as imgutil

def median_filter(img : np.array, maskh : int, maskv : int, extzero=True):
    return imgutil.round_image_colors(imgutil.apply_mask_func_each_channel(np.median, img, maskh, maskv, ext_zero=extzero))
