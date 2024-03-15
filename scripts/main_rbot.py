import cv2  # type: ignore
import numpy as np
import torch
import torch.nn

from segment_anything import build_sam, build_sam_vit_b, build_sam_vit_l, build_sam_vit_h, SamPredictor

import argparse
import json
from typing import Any, Dict, List
from scipy.special import expit

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main_one_img():

    # load image
    imgPath = "/media/ljc/LJC/LJ.jpg"
    img = cv2.imread(imgPath, 1);

    sam_checkpoint = '../models/sam_vit_b_01ec64.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(build_sam_vit_b(checkpoint=sam_checkpoint).to(device))
    print("load sam done.")

    # set image to sam
    sam_predictor.set_image(img)

    # set box
    pt1 = (520, 200)
    pt2 = (620, 300)
    img_rect = img.copy()
    cv2.rectangle(img_rect, pt1, pt2, (0, 0, 255), 2)

    # to sam
    box = np.array([pt1[0], pt1[1], pt2[0], pt2[0]])
    transformed_box = sam_predictor.transform.apply_boxes(box, img.shape[:2])
    in_box = torch.as_tensor(transformed_box, device=sam_predictor.device)

    binary = True
    if binary == True:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=in_box,
            multimask_output=False,
            return_logits=False,
        )

    shape_1 = masks.shape[1]
    image_mask = masks[0][0].cpu().numpy()
    image_mask = (image_mask * 255).astype(np.uint8)

    cv2.imshow("img_rect", img_rect)
    cv2.imshow("image_mask", image_mask)
    cv2.imwrite('/media/ljc/LJC/LJ_mask.jpg', image_mask)
    cv2.waitKey(0)


def main_rbot():

    # load image
    imgPath = "/data/DATASETS/RBOT_dataset_2/ape/frames/a_regular0000.png"
    img = cv2.imread(imgPath, 1);

    # set SAM
    sam_checkpoint = '../models/sam_vit_b_01ec64.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(build_sam_vit_b(checkpoint=sam_checkpoint).to(device))
    print("load sam done.")

    # set image to sam
    sam_predictor.set_image(img)

    # segmentation
    H, W, _ = img.shape

    use_point = True
    use_box = ~use_point

    if use_point:
        point_xy = np.array([[320, 256]])
        transformed_point = sam_predictor.transform.apply_coords(point_xy, img.shape[:2])
        in_points = torch.as_tensor(transformed_point, device=sam_predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        binary = False
        if binary == True:
            masks, iou_preds, lowres_masks = sam_predictor.predict_torch(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=True,
                return_logits=False,
            )
        else:
            masks, iou_preds, lowres_masks = sam_predictor.predict_torch(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=True,
                return_logits=True,
            )
            masks = torch.sigmoid(masks)
            lowres_masks = torch.sigmoid(lowres_masks)

    if use_box:
        box = np.array([270, 178, 270+148, 178+168])
        transformed_box = sam_predictor.transform.apply_boxes(box, img.shape[:2])
        in_box = torch.as_tensor(transformed_box, device=sam_predictor.device)

        binary = False
        if binary == True:
            masks, iou_preds, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=in_box,
                multimask_output=True,
                return_logits=false,
            )
        else:
            masks, iou_preds, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=in_box,
                multimask_output=True,
                return_logits=True,
            )
            masks = torch.sigmoid(masks)

    # get the mask
    shape_1 = masks.shape[1]
    image_mask = masks[0][0].cpu().numpy()
    image_mask = (image_mask*255).astype(np.uint8)

    image_lowres_masks = lowres_masks[0][0].cpu().numpy()
    image_lowres_masks = (image_lowres_masks * 255).astype(np.uint8)

    cv2.imshow("image_mask", image_mask);
    cv2.imshow("image_lowres_mask", image_lowres_masks);
    cv2.waitKey(0)


if __name__ == '__main__':
    main_one_img()
