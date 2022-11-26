import os
import random
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read, read_kitti_flow


class Sintel_Clean(Dataset):
    def __init__(self, root_dir='/home/ltkong/Datasets/MPI-Sintel/training'):
        self.img1_list = []
        self.img2_list = []
        self.flow_list = []
        source_dir = os.path.join(root_dir, 'clean')
        target_dir = os.path.join(root_dir, 'flow')
        for sequence_id in os.listdir(source_dir):
            sequence_dir = os.path.join(source_dir, sequence_id)
            imgs_list = sorted(glob(os.path.join(sequence_dir, '*.png')))
            for i in range(len(imgs_list)-1):
                file_name = imgs_list[i].split('.')[-2].split('/')[-1]
                self.img1_list.append(imgs_list[i])
                self.img2_list.append(imgs_list[i+1])
                self.flow_list.append(os.path.join(target_dir, sequence_id, file_name+'.flo'))
        assert len(self.img1_list) == len(self.img2_list)
        assert len(self.img1_list) == len(self.flow_list)
        self.length = len(self.img1_list)
        print('Found {} Image Pairs'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img1 = read(self.img1_list[idx])
        img2 = read(self.img2_list[idx])
        flow = read(self.flow_list[idx])
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        return img1, img2, flow


class KITTI_2015(Dataset):
    def __init__(self, root_dir='/home/ltkong/Datasets/KITTI/Optical_Flow_Evaluation_2015/training'):
        self.img1_list = sorted(glob(os.path.join(root_dir, 'image_2/*_10.png')))
        self.img2_list = sorted(glob(os.path.join(root_dir, 'image_2/*_11.png')))
        self.flow_list = sorted(glob(os.path.join(root_dir, 'flow_occ/*.png')))
        assert len(self.img1_list) == len(self.img2_list)
        assert len(self.img1_list) == len(self.flow_list)
        self.length = len(self.img1_list)
        print('Found {} Image Pairs'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img1 = read(self.img1_list[idx])
        img2 = read(self.img2_list[idx])
        flow = read_kitti_flow(self.flow_list[idx])
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img2 = torch.from_numpy(img2.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        return img1, img2, flow
