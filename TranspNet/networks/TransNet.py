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


# set defaut configs
configs_file_path = "../configs/paras_train.json"
with open(configs_file_path, 'r') as f:
    configs = json.load(f)

class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()


    def forward(self, x, mask_prev):

        return 0


    def show_model(self):
        model = TransNet()
        print("show the model.")
        for name, parameters in model.named_parameters():
            print(name, ":", parameters.size())


def train(net, train_loader):
    net.train()

    for iter, data in enumerate(train_loader):
        imgs, masks, depths = [d.cuda() for d in data]



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
    train_datas = TOD_Dataset(train=True)
    valid_datas = TOD_Dataset(train=False)

    train_loader = DataLoader(dataset=train_datas, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datas, batch_size=batch_size, shuffle=True)

    train(net, train_loader)




def test_on_RBOT(obj_cur, cur_date):

    # dataset root_path
    root_path = configs['tod path']





