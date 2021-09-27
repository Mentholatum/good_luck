import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    此类用于 PyTorch DataLoader 中创建批处理
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        data_folder: folder where data files are stored 存储数据文件的文件夹
        data_name: base name of processed datasets 已处理数据集的基本名称
        split: split, one of 'TRAIN', 'VAL', or 'TEST' 对'TRAIN', 'VAL', or 'TEST'进行切片
        transform: image transform pipeline 图像变换管道
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored 打开存储图像的hdf5文件
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image 每幅图的标题
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory) 加载编码标题
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory) 加载标题长度
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        #PyTorch transformation pipeline for the image (normalizing, etc.)图像的 PyTorch 转换管道（标准化等）

        self.transform = transform

        # Total number of datapoints 数据点总数
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # the Nth caption corresponds to the (N // captions_per_image)th image
        #第N个标题对应第N个图像
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            #验证测试，返回所有“captions_per_image”标题以找到 BLEU-4 分数
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
