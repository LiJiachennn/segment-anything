import cv2  # type: ignore
import numpy as np
import torch
import torch.nn

from segment_anything import build_sam, SamPredictor

import argparse
import json
from typing import Any, Dict, List
from scipy.special import expit

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():

    # load image
    imgPath = "/data/DATASETS/RBOT_dataset_2/ape/frames/a_regular0000.png"
    img = cv2.imread(imgPath, 1);

    # set SAM
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam_checkpoint = '../models/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    print("load sam done.")

    # set image to sam
    sam_predictor.set_image(img)

    # segmentation
    H, W, _ = img.shape
    point_xy = np.array([[320, 256]])

    transformed_point = sam_predictor.transform.apply_coords(point_xy, img.shape[:2])
    in_points = torch.as_tensor(transformed_point, device=sam_predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

    binary = True
    if binary == True:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            return_logits=False,
        )
    else:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            return_logits=True,
        )
        masks = torch.sigmoid(masks)

    # get the mask
    image_mask = masks[0][0].cpu().numpy()
    image_mask = (image_mask*255).astype(np.uint8)

    cv2.imshow("image_mask", image_mask);
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
