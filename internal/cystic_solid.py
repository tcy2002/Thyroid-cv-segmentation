import cv2
import numpy as np


class CysticSolidDetector:
    def __init__(self):
        pass

    def _preprocess(self, image, thyroid, nodule):
        """
        预处理
        """
        image = cv2.medianBlur(image, 5)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thyroid_gray = cv2.bitwise_and(image_gray, cv2.cvtColor(thyroid, cv2.COLOR_BGR2GRAY))
        nodule_gray = cv2.bitwise_and(image_gray, cv2.cvtColor(nodule, cv2.COLOR_BGR2GRAY))
        return image_gray, thyroid_gray, nodule_gray

    def cystic_solid_detect(self, image_path, thyroid_mask_path, nodule_mask_path):
        """
        判断结节是囊性还是实性
        """
        image = cv2.imread(image_path)
        thyroid = cv2.imread(thyroid_mask_path)
        nodule = cv2.imread(nodule_mask_path)

        # 预处理
        image, thyroid, nodule = self._preprocess(image, thyroid, nodule)

        # 计算结节平均灰度与甲状腺平均灰度的比值
        thyroid_mean = np.sum(thyroid) / np.sum(thyroid > 0)
        nodule_mean = np.sum(nodule) / np.sum(nodule > 0)
        ratio = nodule_mean / thyroid_mean

        # 分级：0-0.3: 纯囊性 0.3-0.6: 稠厚囊性 0.6-0.9: 实性（低回声） 0.85-1.1: 实性（等回声） >1.1: 实性（高回声）
        print(ratio)
        if ratio < 0.3:
            return '纯囊性'
        elif ratio < 0.6:
            return '稠厚囊性'
        elif ratio < 0.9:
            return '实性（低回声）'
        elif ratio < 1.1:
            return '实性（等回声）'
        else:
            return '实性（高回声）'