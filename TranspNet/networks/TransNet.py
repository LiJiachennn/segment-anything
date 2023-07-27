import cv2
import json
import torch
import os
import time
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

from TranspNet.datasets.TOD_dataset import TOD_Dataset
from TranspNet.utils.net_utils import adjust_learning_rate, save_model, save_log

from segment_anything import build_sam, build_sam_vit_b, SamPredictor
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.mask_decoder import MLP


# set defaut configs
configs_file_path = "../configs/paras_train.json"
with open(configs_file_path, 'r') as f:
    configs = json.load(f)

class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()

        sam_checkpoint = '/data/codes/segment-anything/models/sam_vit_b_01ec64.pth'
        sam_predictor = SamPredictor(build_sam_vit_b(checkpoint=sam_checkpoint).cuda())
        self.sam_predictor = sam_predictor

        # shared decoder
        self.c0_upscaling = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            LayerNorm2d(64),
            nn.GELU(),
        )

        # mask decoder
        self.mask_convs_upscaling = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            LayerNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 2, 1, 1)
        )

        # depth decoder
        self.depth_convs_upscaling = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            LayerNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            LayerNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1, 1)
        )

    def forward(self, x):

        # use same to extrackt backbone feature
        feature = self.sam_predictor.extract_image_feature(x)

        # common branch
        c0 = self.c0_upscaling(feature)

        # mask branch
        c_mask = self.mask_convs_upscaling(c0)

        # depth branch
        c_depth = self.depth_convs_upscaling(c0)

        return c_mask, c_depth


    def show_model(self):
        print("show the model.")
        model = TransNet()
        for name, parameters in model.named_parameters():
            print(name, ":", parameters.size())


def train(net, train_loader):
    net.train()

    for iter, data in enumerate(train_loader):
        imgs, masks, depths = [d.cuda() for d in data]

        mask_pred, depth_pred = net.forward(imgs)




def valid(net):
    net.eval()


def train_net(cur_date):

    # globle paras
    epochs = configs['epochs']
    batch_size = configs['batch_size']

    # set network
    gpu_num = torch.cuda.device_count()
    net = TransNet()
    net.show_model()
    net = nn.DataParallel(net, list(range(gpu_num))).cuda()

    # set optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=configs['lr'])

    # set detas
    print("load train data.")
    train_datas = TOD_Dataset(train=True)
    print("load valid data.")
    valid_datas = TOD_Dataset(train=False)

    train_loader = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datas, batch_size=batch_size, shuffle=True)

    train(net, train_loader)




def test_on_RBOT(obj_cur, cur_date):

    # dataset root_path
    root_path = configs['tod path']





