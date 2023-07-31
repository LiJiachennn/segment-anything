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
from TranspNet.utils.net_utils import adjust_learning_rate, save_model, save_log, convert_pred_img_to_origin

from segment_anything import build_sam, build_sam_vit_b, SamPredictor
from segment_anything.modeling.common import LayerNorm2d


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
        self.common_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            LayerNorm2d(256),
            nn.GELU(),
        )
        self.common_upscaling = nn.Sequential(
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
        c0 = self.common_conv(feature)
        c1 = self.common_upscaling(c0)

        # mask branch
        c_mask = self.mask_convs_upscaling(c1)
        c_mask = torch.sigmoid(c_mask)

        # depth branch
        c_depth = self.depth_convs_upscaling(c1)

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
        loss_depth = losses[1](depth_pred, depths)

        loss = losses_weights[0] * loss_mask + losses_weights[1] * loss_depth
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # progress_bar.set_description(f"Batch {iter+1}/{len(train_loader)}")
        # progress_bar.update()

    # save pred img
    training_result_save_dir = '../result/' + 'training_show/'
    if os.path.exists(training_result_save_dir) == False:
        os.mkdir(training_result_save_dir)
    mask_pred_, depth_pred_ = convert_pred_img_to_origin(mask_pred[0:1], depth_pred[0:1])

    mask_pred_save_path = training_result_save_dir + str(epoch).zfill(6) + "_pred_mask.png"
    cv2.imwrite(mask_pred_save_path, mask_pred_)
    depth_pred_save_path = training_result_save_dir + str(epoch).zfill(6) + "_pred_depth.png"
    cv2.imwrite(depth_pred_save_path, depth_pred_)

    # save log
    coef = (1024 * 1024) / (torch.sum(masks[:,0,:,:]).detach().cpu().numpy() / masks.shape[0])
    loss_ = loss.detach().cpu().numpy() * coef
    loss_mask_ = loss_mask.detach().cpu().numpy() * coef
    loss_depth_ = loss_depth.detach().cpu().numpy() * coef
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
        loss_depth = losses[1](depth_pred, depths)
        loss = losses_weights[0] * loss_mask + losses_weights[1] * loss_depth

        coef = (1024 * 1024) / (torch.sum(masks[:,0,:,:]).detach().cpu().numpy() / masks.shape[0])
        loss_mean = loss_mean + loss.detach().cpu().numpy() * coef / (len(valid_loader))
        loss_mask_mean = loss_mask_mean + loss_mask.detach().cpu().numpy() * coef / (len(valid_loader))
        loss_depth_mean = loss_depth_mean + loss_depth.detach().cpu().numpy() * coef / (len(valid_loader))

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
    train_datas = TOD_Dataset(train=True, valid=False, test=False)
    print("load valid data.")
    valid_datas = TOD_Dataset(train=False, valid=True, test=False)

    train_loader = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datas, batch_size=1, shuffle=False)

    # training...
    for epoch in range(epochs):
        train(net, train_loader, optimizer, losses, losses_weights, epoch, log_save_dir)

        loss = valid(net, valid_loader, optimizer, losses, losses_weights, log_save_dir)
        adjust_learning_rate(optimizer, epoch, configs['lr_decay_rate'], configs['lr_decay_epoch'])

        if (epoch+1)%configs['save_epoch_step'] == 0 and bool(configs['save_model']):
            save_model(net, optimizer, epoch, model_save_dir)


def test_net(cur_date, trained_model_index):

    # dataset root_path
    root_path = configs['tod path']

    # load trained model
    gpu_num = torch.cuda.device_count()
    net = TransNet()
    net = nn.DataParallel(net, list(range(gpu_num))).cuda()

    save_dir = configs['TransNet save path'] + cur_date + '/'
    model_save_dir = save_dir + trained_model_index + '.pth'
    state_dict_load = torch.load(model_save_dir)
    net.load_state_dict(state_dict_load['net'])
    net.eval()

    # the seq of TOD dataset
    model_name = ['mug_0', 'mug_1', 'mug_2', 'mug_3', 'mug_4', 'mug_5', 'mug_6']
    texture_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seq_name = [0, 1, 2, 3]
    # model_name = ['mug_0']
    # texture_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # seq_name = [3]

    for m in range(len(model_name)):
        model = model_name[m]
        texture = texture_name[5]
        for s in range(len(seq_name)):
            seq = seq_name[s]
            texture_seq = 'texture_' + str(texture) + '_pose_' + str(seq)
            print(model, texture_seq)

            # load data
            print("load test data.")
            test_datas = TOD_Dataset(train=False, valid=False, test=True,
                                     test_model=model,
                                     test_texture=texture,
                                     test_pose_seq=seq)
            test_loader = DataLoader(dataset=test_datas, batch_size=1, shuffle=False)

            # save path of the pred mask and depth
            pred_maps_save_dir = save_dir + model + "/"
            if os.path.exists(pred_maps_save_dir) == False:
                os.mkdir(pred_maps_save_dir)
            pred_maps_save_dir = pred_maps_save_dir + texture_seq + "/"
            if os.path.exists(pred_maps_save_dir) == False:
                os.mkdir(pred_maps_save_dir)

            # loop
            for iter, data in tqdm(enumerate(test_loader)):
                imgs, _, _ = [d.cuda() for d in data]

                # pred mask and depth
                mask_pred_, depth_pred_ = net.forward(imgs)
                mask_pred, depth_pred = convert_pred_img_to_origin(mask_pred_, depth_pred_)

                mask_pred_save_path = pred_maps_save_dir + str(iter).zfill(6) + "_pred_mask.png"
                cv2.imwrite(mask_pred_save_path, mask_pred)
                depth_pred_save_path = pred_maps_save_dir + str(iter).zfill(6) + "_pred_depth.png"
                cv2.imwrite(depth_pred_save_path, depth_pred)

                cv2.imshow("mask_pred", mask_pred)
                cv2.imshow("depth_pred", depth_pred*30)
                cv2.waitKey(1)