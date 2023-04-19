import cv2
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from segment_anything import build_sam, build_sam_vit_b, build_sam_vit_l, build_sam_vit_h, SamPredictor

"""
Interface methods
"""
def show_img(img):
    return

def cv_version():
    print("python opencv version: ", cv2.getVersionString())

def torch_version():
    print("Pytorch version: ", torch.__version__)

def load_sam_model():
    # load trained model
    # sam_checkpoint = '/data/codes/segment-anything/models/sam_vit_h_4b8939.pth'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    # sam_checkpoint = '/data/codes/segment-anything/models/sam_vit_l_0b3195.pth'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # sam_predictor = SamPredictor(build_sam_vit_l(checkpoint=sam_checkpoint).to(device))

    sam_checkpoint = '/data/codes/segment-anything/models/sam_vit_b_01ec64.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(build_sam_vit_b(checkpoint=sam_checkpoint).to(device))

    print("load model done. model path: ", sam_checkpoint)
    return sam_predictor

def predict_mask_point(sam_predictor, img, point, label, binary):

    # set image to sam
    sam_predictor.set_image(img)

    transformed_point = sam_predictor.transform.apply_coords(point, img.shape[:2])
    in_points = torch.as_tensor(transformed_point, device=sam_predictor.device)
    in_labels = torch.as_tensor(label, device=in_points.device)

    if binary == True:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=False,
        )
    else:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )
        masks = torch.sigmoid(masks)

    # get the mask
    shape_1 = masks.shape[1]
    selectLayer = max(0, shape_1-2)

    image_mask = masks[0][shape_1-1].cpu().numpy()
    image_mask = (image_mask*255).astype(np.uint8)

    return image_mask


def predict_mask_box(sam_predictor, img, box, binary):

    # set image to sam
    sam_predictor.set_image(img)

    transformed_box = sam_predictor.transform.apply_boxes(box, img.shape[:2])
    in_box = torch.as_tensor(transformed_box, device=sam_predictor.device)

    if binary == True:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=in_box,
            multimask_output=True,
            return_logits=False,
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
    selectLayer = max(0, shape_1-2)

    image_mask = masks[0][0].cpu().numpy()
    image_mask = (image_mask*255).astype(np.uint8)

    return image_mask

def predict_mask_mask(sam_predictor, img, mask, binary):

    # set image to sam
    sam_predictor.set_image(img)

    in_mask = torch.as_tensor(mask, device=sam_predictor.device)
    in_mask = in_mask.to(torch.float)
    # map [0, 1] --> (0, 1)
    in_mask = in_mask / 256
    in_mask = torch.logit(in_mask + 0.001)
    in_mask = in_mask[None, :, :]

    if binary == True:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=None,
            mask_input=in_mask,
            multimask_output=True,
            return_logits=False,
        )
    else:
        masks, iou_preds, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=None,
            mask_input=in_mask,
            multimask_output=True,
            return_logits=True,
        )
        masks = torch.sigmoid(masks)

    # get the mask
    image_mask = masks[0][0].cpu().numpy()
    image_mask = (image_mask*255).astype(np.uint8)

    return image_mask