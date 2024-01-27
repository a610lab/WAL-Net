import torch
from PIL import Image
from torch.utils.data import Dataset
from myload_data import read_datalist, load_per_data


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, path, transform):
        self.images_paths, self.labels = read_datalist(path)
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        img, mask = load_per_data(self.images_paths[item])
        label = self.labels[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
