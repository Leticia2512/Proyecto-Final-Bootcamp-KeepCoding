
import numpy as np

def get_nrow_init(img):
    """
    Find the first non-blank row in a 3-channel image.

    Scans the image from the top down to find the first row that contains
    non-zero pixel values in any of the color channels.
    """
    nrow = 0
    for nrow in range(img.shape[0]):
        result = (all(img[nrow, :, 0] == 0),
                  all(img[nrow, :, 1] == 0),
                  all(img[nrow, :, 2] == 0))
        if not all(result):
            break
    return nrow


def get_nrow_end(img):
    """
    Find the last non-blank row in a 3-channel image.

    Scans the image from the bottom up to find the last row that contains
    non-zero pixel values in any of the color channels.
    """
    nrow = img.shape[0]
    for nrow in reversed(range(0, img.shape[0])):
        result = (all(img[nrow, :, 0] == 0),
                  all(img[nrow, :, 1] == 0),
                  all(img[nrow, :, 2] == 0))
        if not all(result):
            break
    return nrow


def get_ncol_init(img):
    """
    Find the first non-blank column in a 3-channel image.

    Scans the image from left to right to find the first column that contains
    non-zero pixel values in any of the color channels.
    """
    ncol = 0
    for ncol in range(img.shape[1]):
        result = (all(img[:, ncol, 0] == 0),
                  all(img[:, ncol, 1] == 0),
                  all(img[:, ncol, 2] == 0))
        if not all(result):
            break
    return ncol


def get_ncol_end(img):
    """
    Find the last non-blank column in a 3-channel image.

    Scans the image from right to left to find the last column that contains
    non-zero pixel values in any of the color channels.
    """
    ncol = img.shape[1]
    for ncol in reversed(range(0, img.shape[1])):
        result = (all(img[:, ncol, 0] == 0),
                  all(img[:, ncol, 1] == 0),
                  all(img[:, ncol, 2] == 0))
        if not all(result):
            break
    return ncol


def expand_image(img, nrow_init, nrow_end, ncol_init, ncol_end):
    """
    Expand an image to make it square by adding padding to the shorter dimension.

    Calculates the difference between height and width of the region of interest,
    then adds equal padding to both sides of the shorter dimension to create
    a square image. Padding is added as black (zero) pixels.
    """
    drow = nrow_end - nrow_init
    dcol = ncol_end - ncol_init
    d = drow - dcol
    nadd = abs(d)
    add_before = nadd//2 + nadd % 2
    add_after = nadd//2
    if d > 0:
        img_expanded = np.pad(img, pad_width=(
            (0, 0), (add_before, add_after), (0, 0)), mode='constant')
    elif d < 0:
        img_expanded = np.pad(img, pad_width=(
            (add_before, add_after), (0, 0), (0, 0)), mode='constant')
    else:
        img_expanded = img
    return img_expanded


def square_image(img):
    """
    Crop an image to its content region and expand it to a square shape.
    """
    nrow_init = get_nrow_init(img)
    nrow_end = get_nrow_end(img)
    ncol_init = get_ncol_init(img)
    ncol_end = get_ncol_end(img)
    img_cropped = img[nrow_init:nrow_end, ncol_init:ncol_end, :]
    expanded_image = expand_image(
        img_cropped, nrow_init, nrow_end, ncol_init, ncol_end)
    return expanded_image
