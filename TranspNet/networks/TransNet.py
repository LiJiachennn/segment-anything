import cv2
import json
import torch
import os
import time
import numpy as np
from tqdm import tqdm

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
        c_mask = torch.sigmoid(c_mask)

        # depth branch
        c_depth = self.depth_convs_upscaling(c0)

        return c_mask, c_depth


    def show_model(self):
        print("show the model.")
        model = TransNet()
        for name, parameters in model.named_parameters():
            print(name, ":", parameters.size())


def train(net, train_loader, optimizer, losses, losses_weights, epoch, log_save_dir):

    net.train()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for iter, data in progress_bar:
        imgs, masks, depths = [d.cuda() for d in data]
        mask_pred, depth_pred = net.forward(imgs)

        loss_mask = losses[0](mask_pred, masks)
        loss_depth = losses[1](depth_pred * masks[:, 0:1, :, :], depths * masks[:, 0:1, :, :])

        loss = losses_weights[0] * loss_mask + losses_weights[1] * loss_depth
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # progress_bar.set_description(f"Batch {iter+1}/{len(train_loader)}")
        # progress_bar.update()

    # save log
    loss_ = loss.detach().cpu().numpy()
    loss_mask_ = loss_mask.detach().cpu().numpy()
    loss_depth_ = loss_depth.detach().cpu().numpy()
    lr_ = optimizer.state_dict()['param_groups'][0]['lr']
    train_ = True
    save_log(epoch, loss_, loss_mask_, loss_depth_, lr_, train_, log_save_dir)
    print("epoch: ", epoch, "\n||train loss:", loss_, "||mask loss:", loss_mask_, "||depth loss:", loss_depth_, "||lr:", lr_)


def valid(net, valid_loader, optimizer, losses, losses_weights, log_save_dir):
    net.eval()
    loss_mean = 0
    loss_mask_mean = 0
    loss_depth_mean = 0

    for iter, data in enumerate(valid_loader):
        imgs, masks, depths = [d.cuda() for d in data]
        mask_pred, depth_pred = net.forward(imgs)

        loss_mask = losses[0](mask_pred, masks)
        loss_depth = losses[1](depth_pred * masks[:, 0:1, :, :], depths * masks[:, 0:1, :, :])
        loss = losses_weights[0] * loss_mask + losses_weights[1] * loss_depth

        loss_mean = loss_mean + loss.detach().cpu().numpy() / (len(valid_loader))
        loss_mask_mean = loss_mask_mean + loss_mask.detach().cpu().numpy() / (len(valid_loader))
        loss_depth_mean = loss_depth_mean + loss_depth.detach().cpu().numpy() / (len(valid_loader))

    save_log(0, loss_mean, loss_mask_mean, loss_depth_mean, 0, False, log_save_dir)
    print("||valid loss:", loss_mean, "||mask loss:", loss_mask_mean, "||depth loss:", loss_depth_mean)

    return loss_mean


def train_net(cur_date):

    # globle paras
    epochs = configs['epochs']
    batch_size = configs['batch_size']

    # set network
    gpu_num = torch.cuda.device_count()
    net = TransNet()
    # net.show_model()
    net = nn.DataParallel(net, list(range(gpu_num))).cuda()

    # set optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=configs['lr'])

    # set losses
    loss_mask = nn.BCELoss(reduction='mean')
    loss_depth = nn.L1Loss(reduction='mean')
    losses = [loss_mask, loss_depth]
    losses_weights = [configs['loss_mask_weight'], configs['loss_depth_weight']]

    # save model
    if bool(configs['save_model']):
        model_save_dir = configs['TransNet save path'] + cur_date + "/"
        if os.path.exists(model_save_dir) == False:
            os.mkdir(model_save_dir)
        log_save_dir = model_save_dir + "log.txt"

    # save config file
    with open(model_save_dir + 'config.json', 'w') as f:
        json.dump(configs, f)

    # set detas
    print("load train data.")
    train_datas = TOD_Dataset(train=True)
    print("load valid data.")
    valid_datas = TOD_Dataset(train=False)

    train_loader = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datas, batch_size=1, shuffle=True)

    # training...
    for epoch in range(epochs):
        train(net, train_loader, optimizer, losses, losses_weights, epoch, log_save_dir)

        loss = valid(net, valid_loader, optimizer, losses, losses_weights, log_save_dir)
        adjust_learning_rate(optimizer, epoch, configs['lr_decay_rate'], configs['lr_decay_epoch'])

        if (epoch+1)%configs['save_epoch_step'] == 0 and bool(configs['save_model']):
            save_model(net, optimizer, epoch, model_save_dir)


def test_net(obj_cur, cur_date):

    # dataset root_path
    root_path = configs['tod path']





