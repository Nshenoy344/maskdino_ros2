from curses import meta
import json
from detectron2.structures.instances import Instances
import numpy as np
import cv2
import torch

import copy
from torch import Tensor
import os

import math
import torch.nn.functional as F
from torch import nn

from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer

from tqdm import tqdm
from shapely.geometry import Polygon, box


def clip_rbbox(corners, img_shape):

    rbbox = Polygon(corners)
    image_rect = box(0, 0, img_shape[0], img_shape[1])

    clipped_rbbox = rbbox.intersection(image_rect)
    if clipped_rbbox.area < 0.3 * rbbox.area:
        return None
    else:
        return np.array(clipped_rbbox.exterior.coords)[:-1]


def visualize_annotations(dicts, metadata):
    # https://github.com/facebookresearch/detectron2/blob/master/tools/visualize_data.py
    for dic in dicts:
        img = utils.read_image(dic["file_name"], "BGR")
        print(dic["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        gt_image = visualizer.draw_dataset_dict(dic)

        image = gt_image.get_image()[:, :, ::-1]
        scale = 1000 / np.max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imshow("Annotated image (ESC to quit)", image)
        k = cv2.waitKey(0)

        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            return


def visualize_data_augmentation(train_loader, metadata, image_format="RGB"):
    for train_image_batch in train_loader:
        for train_image in train_image_batch:  
            # Pytorch tensor is in (C, H, W) format
            img = train_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, image_format)

            visualizer = Visualizer(img, metadata=metadata, scale=1)
            target_fields = train_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )  
               
            cv2.imshow("Augmented image (ESC to quit)", vis.get_image()[:, :, ::-1])
            k = cv2.waitKey(0)

            # exit loop if esc is pressed
            if k == 27:
                cv2.destroyAllWindows()
                return 


def visualize_predictions(predictor, dicts, metadata, threshold=0.8):
    for dic in tqdm(dicts):
        img = utils.read_image(dic["file_name"], "BGR")
        outputs = predictor(img)
        instances = filter_instances_with_score(outputs["instances"].to("cpu"), threshold)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION)
        predictions = visualizer.draw_instance_predictions(instances)

        image = predictions.get_image()[:, :, ::-1]
        scale = 1000 / np.max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        cv2.imshow('Predictions (ESC to quit)', image)
        k = cv2.waitKey(0)

        # exit loop if esc is pressed
        if k == 27:
            cv2.destroyAllWindows()
            return


def filter_instances_with_score(instances, threshold):
    filt_inst = Instances(instances.image_size)
    idxs = np.argwhere(instances.scores > threshold)[0]
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst


def filter_instances_with_area(instances, frac):
    filt_inst = Instances(instances.image_size)
    area = instances.image_size[0] * instances.image_size[1]
    idxs = np.argwhere(torch.sum(instances.pred_masks, [1,2]) > frac * area)[0]
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst


def remove_overlap(instances, threshold):
    filt_inst = Instances(instances.image_size)
    masks = [mask for mask in instances.pred_masks]
    scores = [score for score in instances.scores]
    to_remove = []
    for i in range(len(masks)):
        mask_size = torch.sum(masks[i])
        for j in range(len(masks)):
            if i != j and scores[i] < scores[j] and mask_size > 0:
                intersection = torch.bitwise_and(masks[i].bool(), masks[j].bool())
                inter_size = torch.sum(intersection)
                overlap_frac = inter_size / mask_size
                if overlap_frac > threshold:
                    to_remove.append(i)

    if(len(to_remove) > 0):  
        print(len(set(to_remove)))
    idxs = np.delete(np.arange(len(masks)), to_remove)
    filt_inst.pred_masks = instances.pred_masks[idxs] 
    filt_inst.pred_boxes = instances.pred_boxes[idxs] 
    filt_inst.scores = instances.scores[idxs]
    filt_inst.pred_classes = instances.pred_classes[idxs]
    return filt_inst

     
def get_metadata_from_annos_file(annos_file):
    with open(annos_file, "r") as f:
        data = json.load(f)
        classes = [cat["name"] for cat in data["categories"]]
        metadata = {"thing_classes": classes}
    return metadata
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N, layer_share=False):

    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
