import torch
import fiftyone as fo
# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import inference_on_dataset

from train_net import Trainer

IMAGE_DIR_VAL = '/data_private/data/tldd_test'
DATASET_FILENAME_VAL = '/data_private/data/tldd_test/instances_default.json'
MODEL_FINAL_PATH = '/data_private/outputs/2024-03-30_20-58_maskdinoR50timberseg_100epoch_run3/'

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
                
                tlx, tly, w, h = det.bounding_box
                bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
                fo_poly = det.to_polyline()
                # very small polygons must be discarded because shapely cant deal with them
                if len(fo_poly.points[0]) < 4:
                    continue
                poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
                poly = [p for x in poly for p in x]
                
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

dataset_val = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=IMAGE_DIR_VAL,
    labels_path=DATASET_FILENAME_VAL,
)

DatasetCatalog.register("tldd_val", lambda view=dataset_val: get_fiftyone_dicts(view))
MetadataCatalog.get("tldd_val").set(thing_classes=['logs'], evaluator_type="coco")


# Configuration for the model
cfg = get_cfg() 
cfg.set_new_allowed(True)  # Add this line before merging the file 
cfg.merge_from_file(MODEL_FINAL_PATH + "config.yaml")  # Load your model's configuration file
cfg.OUTPUT_DIR = "./output" 
cfg.MODEL.WEIGHTS = MODEL_FINAL_PATH + "model_final.pth"  # Load your model's checkpoint
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # Set threshold for inference
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Choose device
cfg.DATASETS.TEST = ("tldd_val",)
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes

# Create Predictor
predictor = DefaultPredictor(cfg)

# Evaluate Predictions on Test Dataset
evaluator = Trainer.build_evaluator(cfg, cfg.DATASETS.TEST[0])
val_loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
values_dic = inference_on_dataset(predictor.model, val_loader, evaluator)
print(values_dic)

