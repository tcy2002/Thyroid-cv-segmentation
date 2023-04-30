import os
import numpy as np
import pydicom
from PIL import Image


def get_names(path):
    files = os.listdir(path)
    names = [file for file in files]
    return names


def convert_dcm_jpg(name):
    im = pydicom.dcmread(name)
    im = im.pixel_array.astype(float)
    rescaled_image = (np.maximum(im, 0) / im.max()) * 255.0
    final_image = np.uint8(rescaled_image)
    final_image = Image.fromarray(final_image)
    return final_image


if __name__ == '__main__':
    names = get_names('cases/dcms')
    for name in names:
        image = convert_dcm_jpg(os.path.join('cases/dcms', name))
        image = image.resize((1024, 768))
        if name.endswith('.DCM'):
            name = name.replace('.DCM', '.jpg')
        else:
            name += '.jpg'
        print(name)
        image.save(os.path.join('cases/images', name))
