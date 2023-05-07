import cv2
import numpy as np

from .myconfig import *


class WatershedSegmenter:
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

    def _segment(self, img, seeds):
        # 创建标记图像
        markers = np.zeros(img.shape[:2], dtype=np.int32)

        # 设置标记点为不同的颜色
        for i, seed in enumerate(seeds):
            cv2.drawMarker(markers, tuple(seed), (i+1), cv2.MARKER_TILTED_CROSS)

        cv2.watershed(img, markers)
        return markers

    def segmentation(self, seeds, img_path, out_path):
        img = cv2.imread(img_path)
        blur = self._preprocess(img)

        markers = self._segment(blur, seeds)

        # 将每个区域用不同的颜色显示在原始图像上
        segmentation = np.zeros_like(blur)
        segmentation[markers == 1] = 255
        segmentation[markers == 2] = 0

        segmentation = self._postprocess(segmentation)
        segmentation = cv2.bitwise_and(img, segmentation)
        cv2.imwrite(out_path, segmentation)
