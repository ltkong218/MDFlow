import os
import numpy as np
import torch
from models.FastFlowNet import FastFlowNet
from evaluate import test_sintel_clean, test_kitti_2015


fastflownet = FastFlowNet().cuda().eval()
fastflownet.load_state_dict(torch.load('./checkpoints/fastflownet_kitti.pth'))

sintel_clean_epe = test_sintel_clean(fastflownet)
print('Sintel Clean EPE: {:.3f}'.format(sintel_clean_epe))

kitti_2015_epe = test_kitti_2015(fastflownet)
print('KITTI 2015 EPE: {:.3f}'.format(kitti_2015_epe))


fastflownet = FastFlowNet().cuda().eval()
fastflownet.load_state_dict(torch.load('./checkpoints/fastflownet_sintel.pth'))

sintel_clean_epe = test_sintel_clean(fastflownet)
print('Sintel Clean EPE: {:.3f}'.format(sintel_clean_epe))

kitti_2015_epe = test_kitti_2015(fastflownet)
print('KITTI 2015 EPE: {:.3f}'.format(kitti_2015_epe))
