import numpy as np
import torch
import cv2
import datetime
import time
import sys
import os

from TranspNet.networks.TransNet import train_net, test_net


def main():

    cur_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur_date = "2023-08-01 16:00:00"

    # train_net(cur_date)

    trained_model_index = '0003'
    test_net(cur_date, trained_model_index)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()