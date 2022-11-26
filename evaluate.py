import os
import numpy as np
import torch
import torch.nn.functional as F
from utils import centralize, resize_img, resize_flow, EPE
from datasets import Sintel_Clean, KITTI_2015


div_flow = 20.0

sintel_clean_dataset = Sintel_Clean()
kitti_2015_dataset = KITTI_2015()


def test_sintel_clean(model):
    print('\nTesting Sintel Clean')
    epe_all = 0.0
    model.eval()
    test_iters = len(sintel_clean_dataset)
    for i in range(test_iters):
        img1, img2, flow = sintel_clean_dataset[i]
        img1 = img1.unsqueeze(0).cuda()
        img2 = img2.unsqueeze(0).cuda()
        flow = flow.unsqueeze(0).cuda()
        mask = torch.ones_like(flow[:, :1, :, :]).cuda()

        img1 = resize_img(img1, size=(448, 1024))
        img2 = resize_img(img2, size=(448, 1024))

        img1, img2, _ = centralize(img1, img2)
        imgs =torch.cat([img1, img2], 1)

        with torch.no_grad():
            output = model(imgs).data

        if model.__class__.__name__ == 'FastFlowNet':
            flow_pred = div_flow * F.interpolate(output, size=(448, 1024), mode='bilinear', align_corners=False)
        elif model.__class__.__name__ == 'RAFT':
            flow_pred = F.interpolate(output, size=(448, 1024), mode='bilinear', align_corners=False)

        flow_pred = resize_flow(flow_pred, size=(436, 1024))

        epe_all += EPE(flow_pred, flow, mask)

    epe_all /= test_iters
    return epe_all


def test_kitti_2015(model):
    print('\nTesting KITTI 2015')
    epe_all = 0.0
    model.eval()
    test_iters = len(kitti_2015_dataset)
    for i in range(test_iters):
        img1, img2, flow = kitti_2015_dataset[i]
        img1 = img1.unsqueeze(0).cuda()
        img2 = img2.unsqueeze(0).cuda()
        flow = flow.unsqueeze(0).cuda()
        mask = flow[:, 2:, :, :]
        flow = flow[:, :2, :, :]

        input_size = img1.shape[2:]

        img1 = resize_img(img1, size=(512, 1024))
        img2 = resize_img(img2, size=(512, 1024))

        img1, img2, _ = centralize(img1, img2)
        imgs =torch.cat([img1, img2], 1)

        with torch.no_grad():
            output = model(imgs).data

        if model.__class__.__name__ == 'FastFlowNet':
            flow_pred = div_flow * F.interpolate(output, size=(512, 1024), mode='bilinear', align_corners=False)
        elif model.__class__.__name__ == 'RAFT':
            flow_pred = F.interpolate(output, size=(512, 1024), mode='bilinear', align_corners=False)

        flow_pred = resize_flow(flow_pred, input_size)

        epe_all += EPE(flow_pred, flow, mask)

    epe_all /= test_iters
    return epe_all
