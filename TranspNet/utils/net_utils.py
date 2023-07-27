import os
import torch
import numpy as np

def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    if ((epoch+1) % lr_decay_epoch) != 0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group['lr']
        param_group['lr'] = param_group['lr'] * lr_decay_rate
        param_group['lr'] = max(param_group['lr'], min_lr)
    print('changing learning rate {:5f} to {:.5f}'.format(lr_before,max(param_group['lr'], min_lr)))


def save_model(net, optimizer, epoch, model_dir):
    torch.save({
        'net': net.state_dict(),
        'optim': optimizer.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{:04d}.pth'.format(epoch)))


def save_log(epoch, loss, loss_mask, loss_depth, lr, train, log_dir):
    log_file = open(log_dir, 'a')

    if train== True:
        log_file.write("epoch: {:04d}\t".format(epoch))
        log_file.write("\n")
        log_file.write("train loss: {:.6f}\t".format(loss))
        log_file.write("train mask loss: {:.6f}\t".format(loss_mask))
        log_file.write("train depth loss: {:.6f}\t".format(loss_depth))
        log_file.write("lr: {:.6f}\t".format(lr))
        log_file.write("\n")
    else:
        log_file.write("valid loss: {:.6f}\t".format(loss))
        log_file.write("train mask loss: {:.6f}\t".format(loss_mask))
        log_file.write("train depth loss: {:.6f}\t".format(loss_depth))
        log_file.write("\n")
        log_file.write("\n")
    log_file.close()

def load_pose(path):
    pose_load = np.loadtxt(path)
    poses = np.zeros((pose_load.shape[0], 4, 4))

    for i in range(pose_load.shape[0]):
        pose = np.eye(4)
        pose[0, 0:3] = pose_load[i][0:3]
        pose[1, 0:3] = pose_load[i][3:6]
        pose[2, 0:3] = pose_load[i][6:9]
        pose[0:3, 3] = pose_load[i][9:12] / 1000.0
        poses[i] = pose

    return poses



