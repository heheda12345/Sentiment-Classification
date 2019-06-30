import torch.utils.data as data
import cv2, sys
import numpy as np
import torch

from os import listdir
from os.path import join
from PIL import Image
import re

class SinaDataset(data.Dataset):
    def __init__(self, article_dir, embedding, embedding_len, padding):
        super(SinaDataset, self).__init__()
        self.embedding_len = embedding_len
        self.embedding = embedding
        self.padding = padding
        self.list = []
        self.statistic = np.zeros(8, dtype=np.int32)
        pattern = re.compile(':(\d*)')   # 查找数字
        f = open(article_dir, encoding='utf-8')
        for st in f.readlines():
            s = st[:-1].split('\t') # 去除最后的\n
            target = [int(x) for x in pattern.findall(s[1])][1:] # 匹配出的首个是总数
            input = s[2]
            self.statistic[torch.max(torch.ShortTensor(target).view(-1,1), dim=0)[1]] += 1
            self.list.append([input, target])


    def __getitem__(self, index):
        words = self.list[index][0].split(" ")
        mat = []
        for x in words:
            if x in self.embedding:
                mat.append(self.embedding[x])
            else:
                mat.append([0]*self.embedding_len) # 如果不存在这个词就视为词向量是0，可能有更好的方法
        if (len(mat) > self.padding):
            mat = mat[:self.padding]
        input = torch.cat([torch.FloatTensor(mat),
                           torch.FloatTensor(np.zeros([self.padding-len(mat), self.embedding_len]))])
        target = torch.ShortTensor(self.list[index][1])
        return input, len(mat), target

    def __len__(self):
        return len(self.list)
