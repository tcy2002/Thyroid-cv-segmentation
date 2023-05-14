from cystic_solid import *
from uniformity import *


if __name__ == '__main__':
    name = '262.png'
    image_path = 'images/' + name
    thyroid_mask_path = 'labels/thyroid/' + name
    nodule_mask_path = 'labels/nodule/' + name

    m = UniformityDetector()
    print(m.uniformity_detect(image_path, nodule_mask_path))