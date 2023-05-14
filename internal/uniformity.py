import math

import cv2
import numpy as np


class UniformityDetector:
    def __init__(self):
        self.dist = None
        pass

    def _preprocess(self, image, nodule):
        """
        预处理
        """
        image = cv2.medianBlur(image, 3)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nodule_mask_mask = cv2.cvtColor(nodule, cv2.COLOR_BGR2GRAY)
        nodule_gray = cv2.bitwise_and(image_gray, cv2.cvtColor(nodule, cv2.COLOR_BGR2GRAY))
        return image_gray, nodule_gray, nodule_mask_mask

    def _check_inside(self, nodule, point):
        """
        检查某点是否在结节内部，且到结节边缘的距离不小于5个像素
        """
        # 距离变换
        if self.dist is None:
            self.dist = cv2.distanceTransform(nodule, cv2.DIST_L1, 5)
            self.dist = np.uint8(self.dist)

        # 检查该点是否在结节内部
        if self.dist[point[1], point[0]] < 5:
            return False
        return True

    def _calc_glcm(self, arr, dx, dy, gray_level=16):
        """
        计算灰度共生矩阵
        """
        glcm = np.zeros((gray_level, gray_level), dtype=np.float32)
        arr = arr / (256 // gray_level)
        for i in range(5 - abs(dy)):
            for j in range(5 - abs(dx)):
                glcm[int(arr[i, j]), int(arr[i + dy, j + dx])] += 1
        return glcm

    def _calc_feature(self, image, point):
        """
        计算某点处图像的纹理特征
        """
        # 用灰度共生矩阵计算以point为中心，边长为5的区域内图像的纹理特征
        glcm_x = self._calc_glcm(image[point[1] - 2:point[1] + 3, point[0] - 2:point[0] + 3], 1, 0)
        glcm_y = self._calc_glcm(image[point[1] - 2:point[1] + 3, point[0] - 2:point[0] + 3], 0, 1)
        return glcm_x, glcm_y

    def uniformity_detect(self, image_path, nodule_mask_path):
        """
        判断结节的均匀性
        """
        image = cv2.imread(image_path)
        nodule = cv2.imread(nodule_mask_path)

        # 预处理
        image_gray, nodule_gray, nodule_mask_gray = self._preprocess(image, nodule)

        # 计算结节内部的纹理特征
        x, y = image.shape[1], image.shape[0]
        glcm_xs = []
        glcm_ys = []
        for i in range(0, x, 5):
            for j in range(0, y, 5):
                if nodule_mask_gray[j, i] > 0 and self._check_inside(nodule_mask_gray, (i, j)):
                    glcm_x, glcm_y = self._calc_feature(nodule_gray, (i, j))
                    glcm_xs.append(glcm_x)
                    glcm_ys.append(glcm_y)

        # 计算标准差
        glcm_xs = np.array(glcm_xs)
        glcm_ys = np.array(glcm_ys)
        std_x = np.std(glcm_xs, axis=0)
        std_y = np.std(glcm_ys, axis=0)
        std = math.sqrt(np.mean(std_x) * np.mean(std_y))

        # 分级：0-0.8：均匀；0.08-0.16：较均匀；0.16-0.3：不均匀；0.3-：非常不均匀
        print(std)
        if std < 0.08:
            return '均匀'
        elif std < 0.16:
            return '较均匀'
        elif std < 0.3:
            return '不均匀'
        else:
            return '非常不均匀'
