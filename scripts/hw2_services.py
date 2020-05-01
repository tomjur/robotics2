#!/usr/bin/env python  

import rospy
import tf
import numpy as np
import math

from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse

class GeometricServices(object):
    def __init__(self):
        self.current_joints = None
        
        # Subscribe to /joint_states topic
        # TODO1

        # Create service callback to tf translations
        self.tf_listener = None # TODO2
        # TODO2

        # init your own kinematic chain offsets
        self.a_i = None # TODO3
        self.alpha_i = None # TODO3
        self.d_i = None # TODO3
        self.nue_i = None # TODO3

        # Create service callback to ee pose
        self.direct_translation = rospy.Service('get_ee_pose', Trigger, self.get_ee_pose_callback)


    def joints_callback(self, msg):
        # TODO1

    def get_tf_ee_callback(self, reqt):
        try:
            trans, rot = None, None  # TODO2
            message = 'translation {}, rotation {}'.format(trans, rot)
            return TriggerResponse(success=True, message=message)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return TriggerResponse(success=False, message='Failed, is TF running?')

    def get_ee_pose_callback(self, reqt):
        _, translation, rotation = self._get_ee_pose(self.current_joints)
        message = 'translation {} rotation {}'.format(translation, rotation)
        return TriggerResponse(success=True, message=message)

    def _get_ee_pose(self, joints):
        # TODO6
        from_base_transform, translation, rotation = None, None, None
        return from_base_transform, translation, rotation

    @staticmethod 
    def _generate_homogeneous_transformation(a, alpha, d, nue):
        # TODO4
        return None

    @staticmethod 
    def _rotation_to_quaternion(r):
        # TODO5
        x = y = z = real = None
        return np.array([x, y, z, real])

    def get_geometric_jacobian(self, joints):
        # TODO8
        return None

    def get_analytical_jacobian(self, joints):
        geometric_jacobian = self.get_geometric_jacobian(joints)
        j_p = geometric_jacobian[:3, :]
        j_o = geometric_jacobian[3:, :]
        # TODO9
        return None

    def compute_inverse_kinematics(self, end_pose, max_iterations, error_threshold, time_step, initial_joints, k=1.):
        # TODO10
        return None

    @staticmethod 
    def _normalize_joints(joints):
        res = [j for j in joints]
        for i in range(len(res)):
            res[i] = res[i] + np.pi
            res[i] = res[i] % (2*np.pi)
            res[i] = res[i] - np.pi
        return np.array(res)



def convert_quanternion_to_zyz(q):
    x, y, z, w = q
    # TODO7
    return [None, None, None] 


def solve_ik(geometric_services):
    end_position = [-0.770, 1.562, 1.050]
    end_zyz = convert_quanternion_to_zyz([0.392, 0.830, 0.337, -0.207])
    end_pose = np.concatenate((end_position, end_zyz), axis=0)
    result = gs.compute_inverse_kinematics(end_pose, max_iterations=10000, error_threshold=0.001, time_step=0.001, initial_joints=[0.1]*6)
    print('ik solution {}'.format(result))


if __name__ == '__main__':
    rospy.init_node('hw2_services_node')
    gs = GeometricServices()
    # solve_ik(gs)
    rospy.spin()
