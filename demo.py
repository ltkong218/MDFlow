import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from utils import read, write, centralize, flow_to_color, get_occ_mask

div_flow = 20.0
div_size = 64

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_sintel.pth'))
# model.load_state_dict(torch.load('./checkpoints/fastflownet_kitti.pth'))

img1_path = './data/frame_0013.png'
img2_path = './data/frame_0014.png'
# img1_path = './data/000063_10.png'
# img2_path = './data/000063_11.png'

img1_np = read(img1_path)
img2_np = read(img2_path)
img1 = torch.from_numpy(img1_np).float().permute(2, 0, 1).unsqueeze(0)/255.0
img2 = torch.from_numpy(img2_np).float().permute(2, 0, 1).unsqueeze(0)/255.0
img1, img2, _ = centralize(img1, img2)

height, width = img1.shape[-2:]
orig_size = (int(height), int(width))

if height % div_size != 0 or width % div_size != 0:
    input_size = (
        int(div_size * np.ceil(height / div_size)), 
        int(div_size * np.ceil(width / div_size))
    )
    img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
    img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
else:
    input_size = orig_size

input_fw = torch.cat([img1, img2], 1).cuda()
input_bw = torch.cat([img2, img1], 1).cuda()
input_t = torch.cat([input_fw, input_bw], 0)

output = model(input_t).data

flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

if input_size != orig_size:
    scale_h = orig_size[0] / input_size[0]
    scale_w = orig_size[1] / input_size[1]
    flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
    flow[:, 0, :, :] *= scale_w
    flow[:, 1, :, :] *= scale_h

flow_fw = flow[:1, :, :, :]
flow_bw = flow[1:, :, :, :]

occ_fw, occ_bw = get_occ_mask(flow_fw, flow_bw)

flow_fw = flow_fw[0].cpu().permute(1, 2, 0).numpy()
flow_bw = flow_bw[0].cpu().permute(1, 2, 0).numpy()

occ_fw = (255.0 * occ_fw[0].cpu().permute(1, 2, 0).repeat(1, 1, 3)).numpy().astype(np.uint8)
occ_bw = (255.0 * occ_bw[0].cpu().permute(1, 2, 0).repeat(1, 1, 3)).numpy().astype(np.uint8)

flow_fw_v = flow_to_color(flow_fw)
flow_bw_v = flow_to_color(flow_bw)

img_r1 = np.concatenate((img1_np, img2_np), 1)
img_r2 = np.concatenate((flow_fw_v, flow_bw_v), 1)
img_r3 = np.concatenate((occ_fw, occ_bw), 1)
img_out = np.concatenate((img_r1, img_r2, img_r3), 0)

write('./data/output.png', img_out)
