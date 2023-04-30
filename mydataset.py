"""
自定义数据集
"""
import torch.utils.data as data
from PIL import Image
import os
import torchvision.transforms.functional as TF


class MyDataset(data.Dataset):
    def __init__(self, img_path, label_path):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = os.listdir(self.img_path)
        self.labels = os.listdir(self.label_path)

    def __getitem__(self, index):
        # 读取图片和标签
        img = Image.open(os.path.join(self.img_path, self.imgs[index]))
        img = img.convert("L")
        label = Image.open(os.path.join(self.label_path, self.labels[index]))
        # 将图片和标签转换为tensor
        img = TF.to_tensor(img)
        label = TF.to_tensor(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
