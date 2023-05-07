import cv2
import numpy as np
import os

from .myconfig import *


class Growth:
    def __init__(self):
        pass

    def _preprocess(self, img):
        """
        预处理：中值滤波
        """
        blur = cv2.medianBlur(img, MEDIAN_KERNEL_SIZE)
        return blur

    def _postprocess(self, img):
        """
        后处理：闭运算、均值滤波
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.blur(img, BLUR_KERNEL_SIZE, borderType=cv2.BORDER_REPLICATE)
        img = cv2.threshold(img, POST_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        return img

    def _is_valid_index(self, x, y):
        """
        判断坐标是否在图像范围内
        """
        return 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]

    def _calc_glcm(self, arr, dx, dy, gray_level=16):
        """
        计算灰度共生矩阵
        """
        glcm = np.zeros((gray_level, gray_level), dtype=np.float32)
        arr = arr / (256 // gray_level)
        for i in range(REGION_SIZE - abs(dy)):
            for j in range(REGION_SIZE - abs(dx)):
                glcm[int(arr[i, j]), int(arr[i + dy, j + dx])] += 1
        return glcm

    def _dist_pixel(self, img, x1, y1, x2, y2):
        """
        计算两个像素的距离
        """
        half = REGION_SIZE // 2
        if not self._is_valid_index(x1 - half, y1 - half) or \
                not self._is_valid_index(x2 - half, y2 - half) or \
                not self._is_valid_index(x1 + half, y1 + half) or \
                not self._is_valid_index(x2 + half, y2 + half):
            return 0, 0
        xs1, ys1 = x1 - half, y1 - half
        xe1, ye1 = x1 + half, y1 + half
        xs2, ys2 = x2 - half, y2 - half
        xe2, ye2 = x2 + half, y2 + half

        glcm0x = self._calc_glcm(img[ys1:ye1+1, xs1:xe1+1], 1, 0)
        glcm0y = self._calc_glcm(img[ys1:ye1+1, xs1:xe1+1], 0, 1)
        glcm1x = self._calc_glcm(img[ys2:ye2+1, xs2:xe2+1], 1, 0)
        glcm1y = self._calc_glcm(img[ys2:ye2+1, xs2:xe2+1], 0, 1)

        sim_x = cv2.compareHist(glcm0x, glcm1x, cv2.HISTCMP_CORREL)
        sim_y = cv2.compareHist(glcm0y, glcm1y, cv2.HISTCMP_CORREL)

        gray = np.mean(img[ys2:ye2+1, xs2:xe2+1])

        return sim_x, sim_y, gray

    def _traverse_adjacent_pixel(self, img, new_img, gray, x, y):
        """
        遍历8邻域像素
        """
        new_markers = []
        adjacent = [(x + j * REGION_SIZE, y + i * REGION_SIZE) for i in range(-1, 2) for j in range(-1, 2)]

        for i, j in adjacent:
            if not self._is_valid_index(i, j) or (i == x and j == y) or new_img[j, i] == 255:
                continue

            # 区域增长条件：两个区域的灰度共生矩阵相似度大于阈值，且新区域的灰度与原区域的灰度差值小于阈值
            s1, s2, g = self._dist_pixel(img, x, y, i, j)
            if s1 > REGION_SIMILARITY_THRESHOLD and \
                    s2 > REGION_SIMILARITY_THRESHOLD and \
                    abs(g - gray) < GRAY_DIFF_THRESHOLD:
                cv2.rectangle(new_img, (i - HALF_REGION_SIZE, j - HALF_REGION_SIZE),
                              (i + HALF_REGION_SIZE, j + HALF_REGION_SIZE), 255, -1)
                new_markers.append((i, j))

        return new_img, new_markers

    def _region_growing(self, img, seeds):
        result = np.zeros_like(img)

        for seed in seeds:
            markers = [seed]
            gray = [np.mean(img[seed[1]-REGION_SIZE:seed[1]+REGION_SIZE+1,
                                 seed[0]-REGION_SIZE:seed[0]+REGION_SIZE+1])]
            new_img = np.zeros_like(img)
            cv2.rectangle(new_img, (seed[0] - HALF_REGION_SIZE, seed[1] - HALF_REGION_SIZE),
                          (seed[0] + HALF_REGION_SIZE, seed[1] + HALF_REGION_SIZE), 255, -1)

            while len(markers) > 0:
                marker = markers.pop()
                new_img, new_markers = self._traverse_adjacent_pixel(img, new_img, gray, marker[0], marker[1])
                markers.extend(new_markers)

            result = cv2.bitwise_or(result, new_img)

        return result

    def grow(self, seeds, img_path, out_path):
        """
        区域生长
        """
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        img = self._preprocess(raw_img)

        img = self._region_growing(img, seeds)

        mask = self._postprocess(img)
        img = cv2.bitwise_and(raw_img, mask)

        cv2.imwrite(out_path, img)

