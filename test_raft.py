import os
import numpy as np
import torch
from models.RAFT import RAFT
from evaluate import test_sintel_clean, test_kitti_2015


raft = RAFT().cuda().eval()
raft.load_state_dict(torch.load('./checkpoints/raft_kitti.pth'))

sintel_clean_epe = test_sintel_clean(raft)
print('Sintel Clean EPE: {:.3f}'.format(sintel_clean_epe))

kitti_2015_epe = test_kitti_2015(raft)
print('KITTI 2015 EPE: {:.3f}'.format(kitti_2015_epe))


raft = RAFT().cuda().eval()
raft.load_state_dict(torch.load('./checkpoints/raft_sintel.pth'))

sintel_clean_epe = test_sintel_clean(raft)
print('Sintel Clean EPE: {:.3f}'.format(sintel_clean_epe))

kitti_2015_epe = test_kitti_2015(raft)
print('KITTI 2015 EPE: {:.3f}'.format(kitti_2015_epe))
