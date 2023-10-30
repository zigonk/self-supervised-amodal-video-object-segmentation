# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import random
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import _mask as coco_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def reduce_tensors(tensor):
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    return reduced_tensor


def average_gradients(model):
    """ average gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

def visualize_mask(masks):
    """ visualize masks in shape [seq_len, 1, h, w] """
    import cv2
    import os
    os.makedirs("vis_masks", exist_ok=True)
    masks = masks.cpu().numpy()
    masks = masks.transpose(0, 2, 3, 1)
    masks = masks * 255
    masks = masks.astype(np.uint8)
    for i in range(masks.shape[0]):
        cv2.imwrite(f"vis_masks/mask_{i}.png", masks[i, :, :, 0])

def update_masks(vm_prediction, amodal_pred, object_id):
    """ update vm_prediction with amodal_pred """
    # Convert to binary mask
    amodal_pred = amodal_pred.squeeze() > 0.5
    amodal_pred = amodal_pred.transpose(1, 2, 0).astype(np.uint8) # [seq_len, h, w] -> [h, w, seq_len]
    # Encode amodal_pred to RLE
    amodal_pred = coco_mask.encode(np.asfortranarray(amodal_pred.astype(np.uint8)))
    amodal_pred = [pred['counts'].decode('utf-8') for pred in amodal_pred]
    vm_prediction[object_id]["segmentations"] = amodal_pred