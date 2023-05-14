from cystic_solid import *
from uniformity import *


if __name__ == '__main__':
    name = '40.png'
    image_path = 'images/' + name
    thyroid_mask_path = 'labels/thyroid/' + name
    nodule_mask_path = 'labels/nodule/' + name

    m = UniformityDetector()
    print(m.uniformity_detect(image_path, nodule_mask_path))

    # m = CysticSolidDetector()
    # print(m.cystic_solid_detect(image_path, thyroid_mask_path, nodule_mask_path))
