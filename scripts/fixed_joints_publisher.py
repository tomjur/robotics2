#!/usr/bin/env python

import numpy as np
import rospy

from sensor_msgs.msg import JointState


class FixedJointsPublisher(object):
    def __init__(self, joint_names, joint_values):
        # TODO1: set the private members for this class

        # TODO2: Initialize the node

        # TODO3: Create publisher for joint_states topic to publish a fixed joints messages to

    def publish_joints(self):
        new_msg = JointState()
        new_msg.header.stamp = rospy.Time.now()
        # TODO4: set the joint names and values in the message

        # TODO5: publish the message


if __name__ == '__main__':
    joint_names = ['base_joint', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3']
    joint_values = [0., 0., 0., 0., 0., 0., ]
    # joint_values = [0.5 * np.pi] * 6
    publisher = FixedJointsPublisher(joint_names, joint_values)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
      # TODO6: invoke the joint publishing method	
      rate.sleep()
    rospy.spin()
