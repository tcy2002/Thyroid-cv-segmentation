import nibabel as nib
import numpy as np
import os
from PIL import Image


def get_names(path):
    files = os.listdir(path)
    names = [file for file in files]
    return names


def convert_nii_jpg(name):
    img = nib.load(name)
    img_arr = img.get_fdata()
    img_arr = np.squeeze(img_arr)
    img_arr = np.rot90(img_arr)
    img_arr = np.flipud(img_arr)
    img_arr = (img_arr / img_arr.max()) * 255.0
    img_arr = np.uint8(img_arr)
    img_arr = Image.fromarray(img_arr)
    return img_arr


if __name__ == '__main__':
    names = get_names('cases/niis')
    for name in names:
        image = convert_nii_jpg(os.path.join('cases/niis', name))
        image = image.resize((1024, 768))
        name = name.replace('.nii.gz', '.jpg')
        print(name)
        image.save(os.path.join('cases/labels', name))
