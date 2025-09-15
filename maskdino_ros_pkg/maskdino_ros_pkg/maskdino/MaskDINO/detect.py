
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer

from inference_utils.utils import filter_instances_with_score
import maskdino


class MaskDinoDetector:

    def __init__(self, config_path, model_path, conf_thresh, class_names):
        self._config_path = config_path    
        self._model_path = model_path
        self._conf_thresh = conf_thresh
        self._class_names = class_names
        self.cfg = self.__setup_cfg()
        self.predictor = DefaultPredictor(self.cfg)

    def __setup_cfg(self):
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(self._config_path)
        cfg.MODEL.WEIGHTS = self._model_path
        return cfg
    
    def detect(self, image):
        outputs = self.predictor(image)
        instances = filter_instances_with_score(outputs["instances"].to("cpu"), self._conf_thresh)
        # Step 4: Extract the labels and visualize
        # TODO Find solution to replace the thins_classes
        visualizer = Visualizer(image[:, :, ::-1], {"thing_classes":self._class_names}, scale=1, instance_mode=ColorMode.SEGMENTATION)
        predictions = visualizer.draw_instance_predictions(instances)
        image = predictions.get_image()[:, :, ::-1]

        return instances, image