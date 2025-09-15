import torch
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, sys, shutil, json, random, cv2
import datetime
import copy
from itertools import groupby

import fiftyone as fo
import fiftyone.utils.random as four

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets.coco import convert_to_coco_json
#from detectron2.data.datasets import register_coco_instances
# setup and launch the trainer
from train_net import Trainer
import maskdino
import warnings

#########################
### PROGRAM VARIABLES ###
#########################

DO_TRAIN = True              # Whether to train the model
DO_TEST = True               # Whether to test the model 
RESUME = False               # If True, resumes from last_checkpoint

DATASET_FILENAME_TRAIN = '/data_private/data/tldd_train/instances_prescaled.json'          # COCO style annotation file
IMAGE_DIR_TRAIN = '/data_private/data/tldd_train'                                    # Folder where dataset was downloaded
DATASET_FILENAME_VAL = '/data_private/data/tldd_val/instances_prescaled.json'          # COCO style annotation file
IMAGE_DIR_VAL = '/data_private/data/tldd_val'
SPLIT_FRAC = 0.8                                                  # Fraction of data used in train
OUTPUT_DIR = "/data_private/outputs"                               # Folder where to save outputs

CONFIG_FILE = "configs/config_maskdino_swinB.yaml"      # Detectron2 style config file
TEST_WEIGHTS = ""                                       # Checkpoint to use for testing
INITIAL_WEIGHTS = None                                # Path to a previous checkpoint to finetune or None

SHOW_ANNOTATIONS = False        # Display images and labels before training
SHOW_AUGMENTATIONS = False      # Display augmented images and labels before training


warnings.filterwarnings("ignore", category=RuntimeWarning) 

def place_custom_mask_in_array(custom_mask, new_array_size, position):
    """
    Place a custom shape mask into a new array.

    :param custom_mask: 2D array representing the custom shape mask
    :param new_array_size: Size of the new array (height, width)
    :param position: Position to place the mask in the new array (x, y)
    :return: New array with the custom mask placed
    """
    new_array = np.zeros(new_array_size, dtype=bool)
    
    x, y = position
    mask_height, mask_width = custom_mask.shape

    # Ensure the mask fits in the new array
    if y + mask_height > new_array_size[0] or x + mask_width > new_array_size[1]:
        print("Position: ", position)
        print("New array size: ", new_array_size)
        print("Size of custom mask: ", mask_width, mask_height)
        print("End coordinates of custom mask: ", x+mask_width, y+mask_height)
        raise ValueError("Custom mask exceeds new array boundaries")

    new_array[y:y+mask_height, x:x+mask_width] = custom_mask

    return new_array

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def get_fiftyone_dicts(samples):
    samples.compute_metadata()
    dataset_dicts = []
    for sample in samples.select_fields(["id", "filepath", "metadata", "segmentations"]):
        if sample.segmentations is not None:
            height = sample.metadata["height"]
            width = sample.metadata["width"]
            record = {}
            record["file_name"] = sample.filepath
            record["image_id"] = sample.id
            record["height"] = height
            record["width"] = width

            objs = []
            for det in sample.segmentations.detections:
                #print(det)
                tlx, tly, w, h = det.bounding_box
                bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
                fo_poly = det.to_polyline()
                # very small polygons must be discarded because shapely cant deal with them
                if len(fo_poly.points[0]) < 4:
                    continue
                poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
                poly = [p for x in poly for p in x]
                #print(det.label)
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    #"gt_masks": det.mask
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def get_metadata_from_annos_file(annos_file):
    with open(annos_file, "r") as f:
        data = json.load(f)
        classes = [cat["name"] for cat in data["categories"]]
        metadata = {"thing_classes": classes}
    return metadata

def init_mask_dino(config_file, output_dir, fold_num=None, initial_weights=None): 

    # Configure Model - See all parameters here : https://detectron2.readthedocs.io/en/latest/modules/config.html#yaml-config-references
    cfg = get_cfg()
    maskdino.add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = output_dir

    # Initialize network weights (to finetune)
    if initial_weights is not None:
        cfg.MODEL.WEIGHTS = initial_weights

    DatasetCatalog.register("tldd_train", lambda view=dataset_train: get_fiftyone_dicts(view))
    MetadataCatalog.get("tldd_train").set(thing_classes=['logs'], evaluator_type="coco")

    DatasetCatalog.register("tldd_val", lambda view=dataset_val: get_fiftyone_dicts(view))
    MetadataCatalog.get("tldd_val").set(thing_classes=['logs'], evaluator_type="coco")

    # Save Model config in output folder
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/config.yaml", "w") as f:
        f.write(cfg.dump())

    return cfg

def train_mask_dino(cfg, resume=False):
    
    # Create Trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume)

    # Train
    trainer.train()


if __name__ == "__main__":

    print('GPU available :', torch.cuda.is_available())
    print('Torch version :', torch.__version__, '\n')

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Init output folder 
    if (RESUME):
        # Take latest run in outputs folder
        output_dir = np.sort([x for x in os.listdir("./outputs/") if os.path.isdir("./outputs/"+x)])[-1]
        output_dir = os.path.join("./outputs", output_dir)
        config_file = os.path.join(output_dir, "config.yaml")
    else:
        output_dir = OUTPUT_DIR + f'/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'

    #random_state = np.random.RandomState(42)
    #dataset_dicts = np.array(load_coco_json(DATASET_FILENAME, IMAGE_DIR))

    # Import the dataset
    dataset_train = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=IMAGE_DIR_TRAIN,
        labels_path=DATASET_FILENAME_TRAIN,
    )

    dataset_val = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=IMAGE_DIR_VAL,
        labels_path=DATASET_FILENAME_VAL,
    )

    cfg = init_mask_dino(CONFIG_FILE, output_dir, initial_weights=INITIAL_WEIGHTS)

    if DO_TRAIN:
        train_mask_dino(cfg, RESUME)
        TEST_WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")       # Ensures trained network will be used for testing 

    if DO_TEST:
        pass
