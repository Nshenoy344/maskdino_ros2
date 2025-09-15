#!/usr/bin/python3

import rclpy
from tf2_msgs.msg import TFMessage
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose

# Inspired by project ICP object fitting from @knofm 
 
class Transformer():
    """ 
    Handles transformations from poses and frames into other frames.
    """
    def __init__(self,node,tf_message):
        """
        Initialize subscriber of topic tf.
        """
        #rospy.Subscriber('/tf', TFMessage, self.tf_cb)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer,node)
        #self.tf_message = TFMessage()
        self.tf_message=tf_message

    def tf_cb(self,msg):
        self.tf_message = msg
    
    def transform_pose(self,pose,source_frame, target_frame):
        """
        Transforms the pose from the sourceFrame into the targetFrame.
        Returns transformation as PoseStamped.
        """
        trans = self.tf_buffer.lookup_transform(target_frame,
                                                source_frame,
                                                Time(),
                                                Duration(seconds=1.0))
        trans_stamped = TransformStamped()
        trans_stamped.transform = trans.transform

        # pose_stamped = PoseStamped()
        # pose_stamped.pose = pose

        #return do_transform_pose(pose_stamped, trans_stamped).pose
        return do_transform_pose(pose, trans_stamped)        
