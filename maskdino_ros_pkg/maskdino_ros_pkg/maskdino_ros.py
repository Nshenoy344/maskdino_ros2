#!/usr/bin/python3

import rclpy
import json
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from maskdino_ros_pkg.utils.utils import Utils

import sys
sys.path.append('/root/ros2_ws/src/maskdino_ros_pkg/maskdino_ros_pkg/maskdino/MaskDINO')
from detect import MaskDinoDetector

class MaskDinoRos(Node):

    def __init__(self):
        super().__init__('maskdino_ros_node')
        self.ns=self.get_name()+ "/"
        self.__declare_parameters()
        self.__get_params()
        self.get_labels()
        self.tf_message = TFMessage()
        self.mask_dino_detector = MaskDinoDetector(conf_thresh=self.conf_thresh, 
                                                   model_path=self.weights_path, 
                                                   config_path=self.config_path, 
                                                   class_names= self.only_label)
        self.image = None
        self.pcl = None
        self.cv_bridge = CvBridge()
        
        self.__init_subs_and_pubs()
        self.id_mapping_msg=String()
        self.utils_cls = Utils(self,self.tf_message)
        self.pose_warning = False
        self.image_warning = False
       
        
        #self.start()

    def __init_subs_and_pubs(self):
        self.image_subscriber=self.create_subscription(Image,self.img_topic, self.__img_cb,10)

        if self.depth_topic != "":
            self.image_depth_sub = self.create_subscription(PointCloud2,self.depth_topic,self.__pcl_cb,10)
        else:
            if not self.pose_warning:
                self.get_logger().warn("No pose estimaton enabled, to get the pose estimation add a depth image topic to the roslaunch command")
                self.pose_warning=True
        self.detection_publisher = self.create_publisher(Detection2DArray,self.out_topic,10)
        vis_topic = self.out_topic + "visualization" if self.out_topic.endswith("/") else self.out_topic + "/visualization"
        
        self.img_publisher = self.create_publisher(Image,vis_topic,10) if self.visualize else None

        self.class_mapping_topic=self.out_topic + "id_to_category" if self.out_topic.endswith("/") else self.out_topic + "/id_to_category"

        self.class_mapping_publisher=self.create_publisher(String,self.class_mapping_topic,10)

        self.tf_subscriber=self.create_subscription(TFMessage,'/tf', self.tf_cb,10)

    def __declare_parameters(self):
        self.declare_parameter('weights_path' ,'')
        self.declare_parameter('img_topic' ,'')
        self.declare_parameter('out_topic' ,'')
        self.declare_parameter('conf_thresh' ,0.5)
        self.declare_parameter('config_path' ,'')
        self.declare_parameter('visualize' ,True)
        self.declare_parameter('depth_topic' ,'')
        self.declare_parameter('target_frame_id' ,'')
        self.declare_parameter('labels_path' ,'')

    def __get_params(self):
        self.weights_path=self.get_parameter('weights_path').get_parameter_value().string_value
        self.img_topic = self.get_parameter('img_topic').get_parameter_value().string_value
        self.out_topic = self.get_parameter('out_topic').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('conf_thresh').get_parameter_value().double_value
        self.config_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.target_frame_id = self.get_parameter('target_frame_id').get_parameter_value().string_value
        self.labels_path=self.get_parameter('labels_path').get_parameter_value().string_value
        self.get_logger().info(f'image topic = {self.img_topic}')

    def get_labels(self):
        self.get_logger().info(f'labels path = {self.labels_path} , weights path = {self.weights_path}')
        print(self.labels_path)
        with open(self.labels_path, 'r') as file:
            data = json.load(file)
            categories = data['categories']
            self.only_label= [category['name'] for category in categories]
            #get the class to id mapping
            self.classes_mapping={}
            for item , category in enumerate(self.only_label):
                self.classes_mapping[item]=category

            
    def __img_cb(self, msg):
        self.get_logger().info(f'image height ={msg.height} ')
        self.image = msg
        self.start()

    def __pcl_cb(self, msg):
        self.pcl = msg

    def tf_cb(self,msg):
        self.tf_message = msg

    def start(self):

        #self.image_subscriber=self.create_subscription(Image,self.img_topic, self.__img_cb,10)


        #while self.image == None:
        if not self.image_warning and self.image==None:
                self.get_logger().warn("No Image message available, please check your topic")
                self.image_warning=True

        np_img_orig = self.cv_bridge.imgmsg_to_cv2(
            self.image, desired_encoding='bgr8'
        )

        instances, image =self.mask_dino_detector.detect(np_img_orig)
        detection_msg = Detection2DArray()
        if len(instances.pred_boxes) > 0:
            detection_msg = self.utils_cls.create_detection_msg(self.image, instances, self.pcl, self.target_frame_id)
        
        self.detection_publisher.publish(detection_msg)

        if self.visualize:        
            vis_msg = self.cv_bridge.cv2_to_imgmsg(image)
            self.img_publisher.publish(vis_msg)
        
        self.id_mapping_msg.data=json.dumps(self.classes_mapping)

        self.class_mapping_publisher.publish(self.id_mapping_msg)



def main():
    rclpy.init()
    mask_dino_ros2 = MaskDinoRos()
    try:
        rclpy.spin(mask_dino_ros2)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()



if __name__ == '__main__':
    main()
