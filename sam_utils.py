import os
import numpy as np
import torch
# import matplotlib.pyplot as plt
import cv2
from PIL import Image
# from PIL import Image
import time
from utils import find_image_file

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def sam_init(device_id=0):
    sam_checkpoint = "/home/chao/chao/OpenComplete/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # mask_generator = SamAutomaticMaskGenerator(sam)
    return predictor

def sam_out(predictor, shape_dir):
    image_path = os.path.join(shape_dir, find_image_file(shape_dir))
    save_path = os.path.join(shape_dir, "image_sam.png")
    bbox_path = os.path.join(shape_dir, "bbox.txt")
    bbox = np.loadtxt(bbox_path, delimiter=',')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    predictor.set_image(image)

    h, w, _ = image.shape
    input_point = np.array([[h//2, w//2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    opt_idx = np.argmax(scores)
    mask = masks[opt_idx]
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image[:, :, 3] = mask.astype(np.uint8) * 255
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255 # np.argmax(scores_bbox)
    cv2.imwrite(save_path, cv2.cvtColor(out_image_bbox, cv2.COLOR_RGBA2BGRA))


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(img)
    # return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return np.asarray(img)
    # return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def sam_out_nosave(predictor, input_image, *bbox_sliders):
    # save_path = os.path.join(shape_dir, "image_sam.png")
    # bbox_path = os.path.join(shape_dir, "bbox.txt")
    # bbox = np.loadtxt(bbox_path, delimiter=',')
    bbox = np.array(bbox_sliders)
    image = convert_from_image_to_cv2(input_image)

    start_time = time.time()
    predictor.set_image(image)

    h, w, _ = image.shape
    input_point = np.array([[h//2, w//2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks_bbox, scores_bbox, logits_bbox = predictor.predict(
        box=bbox,
        multimask_output=True
    )

    print(f"SAM Time: {time.time() - start_time:.3f}s")
    opt_idx = np.argmax(scores)
    mask = masks[opt_idx]
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image[:, :, 3] = mask.astype(np.uint8) * 255
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255 # np.argmax(scores_bbox)
    return Image.fromarray(out_image_bbox, mode='RGBA') 
    cv2.imwrite(save_path, cv2.cvtColor(out_image_bbox, cv2.COLOR_RGBA2BGRA))