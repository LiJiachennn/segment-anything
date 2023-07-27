import torch
from torch.utils.data import DataLoader

import numpy as np
import cv2, os, sys, json, glob

from TranspNet.utils.net_utils import load_pose

# set defaut configs
configs_file_path = "../configs/paras_train.json"
with open(configs_file_path, 'r') as f:
    configs = json.load(f)

class TOD_Dataset(torch.utils.data.Dataset):

    def __init__(self, train=True):
        """
        :param test: for train or test, when training the network
        """
        super(TOD_Dataset, self).__init__()

        # set basic paras
        self.img_w = 1280
        self.img_h = 720

        # set paths
        self.tod_path = configs['tod path']
        self.tod_externel_path = configs['tod externel path']
        self.train = train

        # set models
        # self.model_name = ['mug_0', 'mug_1', 'mug_2', 'mug_3', 'mug_4', 'mug_5', 'mug_6']   # only test on mug model
        # self.test_texture = 5                                                               # same as the keypose method
        self.model_name = ['mug_0']   # only test on mug model
        self.test_texture = 5                                                               # same as the keypose method

        # set textures for train and test split
        if self.train == True:
            # self.textures = [0, 1, 2, 3, 4, 6, 7, 8, 9]
            self.textures = [5]
        if self.train == False:
            self.textures = [5]

        # declare some data paras
        self.img_files = np.array([])
        self.mask_files = np.array([])
        self.depth_files = np.array([])

        # loop the seqs
        for m in range(len(self.model_name)):
            for t in range(len(self.textures)):
                for p in range(4):
                    image_dir = self.tod_path + self.model_name[m] + "/texture_" + str(t) + "_pose_" + str(p) + "/"
                    image_externel_dir = self.tod_externel_path + self.model_name[m] + "/texture_" + str(t) + "_pose_" + str(p) + "/"
                    if not os.path.exists(image_dir):
                        continue

                    # load gt pose, for check the valid depth imgs
                    poses = load_pose(image_externel_dir + "pose_gt.txt")

                    # load files dir
                    filenames_img = glob.glob(os.path.join(image_dir, '*_L.png'))
                    filenames_mask = glob.glob(os.path.join(image_dir, '*_mask.png'))
                    filenames_depth = glob.glob(os.path.join(image_externel_dir, '*_Dr.png'))
                    filenames_img.sort()
                    filenames_mask.sort()
                    filenames_depth.sort()
                    assert poses.shape[0] == len(filenames_img), 'the shape of pose is not equal to the img files'
                    assert poses.shape[0] == len(filenames_mask), 'the shape of pose is not equal to the mask files'
                    assert poses.shape[0] == len(filenames_depth), 'the shape of pose is not equal to the depth files'

                    # loop the current seq
                    for i in range(poses.shape[0]):
                        pose = poses[i]
                        if pose[0][0] == 1.0 and pose[2][3] == 1.0:
                            continue

                        self.img_files = np.append(self.img_files, filenames_img[i])
                        self.mask_files = np.append(self.mask_files, filenames_mask[i])
                        self.depth_files = np.append(self.depth_files, filenames_depth[i])

        print("img_files.shape: ", self.img_files.shape)
        print("mask_files.shape: ", self.mask_files.shape)
        print("depth_files.shape: ", self.depth_files.shape)


    def __getitem__(self, index):

        # load the origin imgs
        img = cv2.imread(self.img_files[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(self.mask_files[index], cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self.depth_files[index], cv2.IMREAD_UNCHANGED)

        # from [0,255] to [0.0, 1.0]
        mask = self.normalize_img(mask)

        # fill the image and resize to [256, 256]
        bottom = self.img_w - self.img_h
        mask_border = cv2.copyMakeBorder(mask, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0))
        depth_border = cv2.copyMakeBorder(depth, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0))

        lowres_size = (1024, 1024)
        img_lowres = cv2.resize(img, (1024, 576), interpolation=cv2.INTER_LINEAR)
        mask_lowres = cv2.resize(mask_border, lowres_size, interpolation=cv2.INTER_LINEAR)
        depth_lowres = cv2.resize(depth_border, lowres_size, interpolation=cv2.INTER_NEAREST)

        # process the mask
        mask_lowres = np.expand_dims(mask_lowres, axis=2)
        mask_lowres = np.concatenate((mask_lowres, mask_lowres), axis=2)
        mask_lowres[:, :, 1] = 1.0 - mask_lowres[:, :, 0]

        # to tensor
        img_lowres = torch.tensor(img_lowres.astype(np.float32)).permute(2, 0, 1)
        mask_lowres = torch.tensor(mask_lowres.astype(np.float32)).permute(2, 0, 1)
        depth_lowres = np.expand_dims(depth_lowres, axis=2)
        depth_lowres = torch.tensor(depth_lowres.astype(np.float32)).permute(2, 0, 1)

        return img_lowres, mask_lowres, depth_lowres


    def __len__(self):
        return self.img_files.shape[0]

    def normalize_img(self, img):
        return img / 255;

