import cv2
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from segment_anything import build_sam, SamPredictor

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
    sam_checkpoint = '/data/codes/segment-anything/models/sam_vit_h_4b8939.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    print("load model done. model path: ", sam_checkpoint)
    return sam_predictor

def predict_mask_point(sam_predictor, img, point, binary):

    # set image to sam
    sam_predictor.set_image(img)

    transformed_point = sam_predictor.transform.apply_coords(point, img.shape[:2])
    in_points = torch.as_tensor(transformed_point, device=sam_predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

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
    image_mask = cv2.convertScaleAbs(image_mask*255).astype(np.uint8)

    return image_mask