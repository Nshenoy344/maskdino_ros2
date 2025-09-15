import rclpy
import math
from rclpy.time import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, Point2D,Pose2D ,  \
    ObjectHypothesis,ObjectHypothesisWithPose
#from geometry_msgs.msg import Pose2D, Pose
from geometry_msgs.msg import Pose
import sensor_msgs_py.point_cloud2 as pc2

#from utils.transformer import Transformer
from maskdino_ros_pkg.utils.transformer import Transformer

class Utils():

    def __init__(self,node,tf_message):
        self.node=node
        self.tf_message=tf_message
        self.transformer = Transformer(self.node,self.tf_message)
        pass

    def create_detection_msg(self, img_msg: Image, instances, image_depth_msg = None, target_frame_id="base_footprint") -> Detection2DArray:
        """
        :param img_msg: original ros image message
        :param detections: torch tensor of shape [num_boxes, 6] where each element is
            [x1, y1, x2, y2, confidence, class_id]
        :returns: detections as a ros message of type Detection2DArray
        """
        detection_array_msg = Detection2DArray()

        # header
        header = self.create_header()
        detection_array_msg.header = header
        detection_array_msg.header.frame_id  = target_frame_id

        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        for idx, bbox in enumerate(instances.pred_boxes.tensor.numpy()):
            single_detection_msg = Detection2D()
            single_detection_msg.header = header
            single_detection_msg.header.frame_id = target_frame_id

            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            # bbox
            bbox_msg = BoundingBox2D()
            # w = int(round(x2 - x1))
            # h = int(round(y2 - y1))
            # cx = int(round(x1 + w / 2))
            # cy = int(round(y1 + h / 2))
            w = float(round(x2 - x1))
            h = float(round(y2 - y1))
            cx = float(round(x1 + w / 2))
            cy = float(round(y1 + h / 2))
            bbox_msg.size_x = w
            bbox_msg.size_y = h

            #old pose using geometry msgs , deprecated in ros2 humble

            # bbox.center = Pose2D()
            # bbox.center.x = cx
            # bbox.center.y = cy

            bbox_msg.center = Pose2D() 
            bbox_msg.center.position= Point2D()
            bbox_msg.center.position.x = cx
            bbox_msg.center.position.y = cy

            single_detection_msg.bbox = bbox_msg

            # class id & confidence
            obj_hyp = ObjectHypothesisWithPose()
            obj_hyp.hypothesis=ObjectHypothesis()
            obj_hyp.hypothesis.class_id=f'{pred_classes[idx]}'
            obj_hyp.hypothesis.score=float(scores[idx])


            # obj_hyp.id = int(pred_classes[idx])
            # obj_hyp.score = scores[idx]
            if image_depth_msg != None:
                obj_hyp.pose.pose = self.transformer.transform_pose(self.estimate_pose(cx,cy, image_depth_msg), img_msg.header.frame_id, target_frame_id) 
                obj_hyp.pose.pose.orientation.x = 0.0
                obj_hyp.pose.pose.orientation.y = 0.0
                obj_hyp.pose.pose.orientation.z = 0.0
                obj_hyp.pose.pose.orientation.w = 1.0

            single_detection_msg.results = [obj_hyp]

            detection_array_msg.detections.append(single_detection_msg)

        return detection_array_msg

    def create_header(self):
        h = Header()
        h.stamp = rclpy.clock.Clock().now().to_msg() 
        return h



    def estimate_pose(self,xc, yc, point_cloud):
    
        # Calculate Pose
        pose = Pose()
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        arrayPos = (int(yc)* point_cloud.width) + int(xc)
        gen = pc2.read_points(point_cloud, skip_nans=False, field_names=("x", "y", "z"))
        gen_list = list(gen)
        arrayPosX = arrayPos
        arrayPosY = arrayPos + 4
        arrayPosZ = arrayPos + 8
        x = gen_list[arrayPosX][0]
        y = gen_list[arrayPosY][1]
        z = gen_list[arrayPosZ][2]

        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.0
        else:
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)

        return pose
