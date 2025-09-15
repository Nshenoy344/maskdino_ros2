import torch
import os, cv2
from tqdm import tqdm
import torch
import os
import json

import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils import logger
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import ColorMode, Visualizer

from inference_utils.utils import filter_instances_with_score, get_metadata_from_annos_file
import maskdino

#########################
### PROGRAM VARIABLES ###
#########################

OUTPUT_FOLDER = "/source/model_store_groceries_v2/" # Training outputs to use for inference
DIRECTORY = '/imgs/test_data'                     				# Directory from which to read images to predict
DETECTION_THRESHOLD = 0.4                              		# Minimal network confidence to keep instance

#########################


if __name__ == "__main__":

	print('GPU available :', torch.cuda.is_available())
	print('Torch version :', torch.__version__, '\n')
	logger = logger.setup_logger(name=__name__)

	# Configure Model
	cfg = get_cfg()
	cfg.set_new_allowed(True)
	cfg.merge_from_file(os.path.join(OUTPUT_FOLDER, "config.yaml"))
	cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_FOLDER, "model_final.pth")
	label_path=os.path.join(OUTPUT_FOLDER, "instances_default.json")


	# Create Predictor
	predictor = DefaultPredictor(cfg)

	#get labels
	with open(label_path, 'r') as file:
		data = json.load(file)
		categories = data['categories']
		label_to_int = {category['name']: category['id'] for category in categories}
		only_label= [category['name'] for category in categories]

	
	# Run inference on selected folder
	count = 0
	for filename in tqdm(os.listdir(DIRECTORY)):
		if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):

			# Load image and metadata
			filepath = os.path.join(DIRECTORY, filename)
			image = utils.read_image(filepath, "BGR")

			# Run network on image
			outputs = predictor(image)
			instances = filter_instances_with_score(outputs["instances"].to("cpu"), DETECTION_THRESHOLD)

			# Visualize image
			visualizer = Visualizer(image[:, :, ::-1],{"thing_classes":only_label}, scale=1, instance_mode=ColorMode.SEGMENTATION)
			predictions = visualizer.draw_instance_predictions(instances)
			
			# Save image
			cv2.imwrite("/outputs/output_storing_v2/frame%d.jpg" % count, predictions.get_image()[:, :, ::-1])
			#cv2.imshow('Predictions (ESC to quit)', predictions.get_image()[:, :, ::-1])
			#k = cv2.waitKey(0)
			count +=1
			# exit loop if esc is pressed
			#if k == 27:
			#	cv2.destroyAllWindows()
			#	break


