import cv2
import numpy as np


images = ['55.png', '219.png', '222.png', '202303081019410072SMP.jpg', 'IM_0186.jpg']


# 使用交叉熵计算两个图像的相似度
def cross_entropy(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 /= 255
    img2 /= 255
    img1[img1 < 0.01] = 0.0001
    img2[img2 < 0.01] = 0.0001
    img1[img1 > 0.99] = 0.9999
    img2[img2 > 0.99] = 0.9999
    return -np.sum(img1 * np.log(img2) + (1 - img1) * np.log(1 - img2)) / img1.size


# 使用dice系数计算两个图像的相似度
def dice(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1 /= 255
    img2 /= 255
    return 2 * np.sum(img1 * img2) / (np.sum(img1) + np.sum(img2))


for name in images:
    mask1 = cv2.imread('growth/' + name, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread('watershed/' + name, cv2.IMREAD_GRAYSCALE)
    mask3 = cv2.imread('unet/' + name, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('mask/' + name, cv2.IMREAD_GRAYSCALE)
    print(name, dice(mask1, mask), dice(mask2, mask), dice(mask3, mask))
