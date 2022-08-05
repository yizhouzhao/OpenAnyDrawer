import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

from hand.helper import HandHelper
from numpy_utils import *
from utils import get_mesh_bboxes

from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.robots.robot_view import RobotView

class HandEnv():
    def __init__(self,
        prim_paths_expr="",
        xform_paths_expr="",
        backend = "numpy",
        device = None
        ) -> None:
        
        # init hand helper
        # self.hander_helper = HandHelper()

        self.xform_paths_expr = xform_paths_expr
        self.prim_paths_expr = prim_paths_expr
        self.backend = backend
        self.device = device

    def start(self): 
        # simulation context
        self.simlation_context = SimulationContext(backend=self.backend, device=self.device)
        print("simlation context", SimulationContext.instance().backend, SimulationContext.instance().device)

        # articulation
        self.robots =  RobotView(self.prim_paths_expr) # sim.create_articulation_view("/World/envs/*/humanoid/torso") # 
        self.robot_indices = self.robots._backend_utils.convert(np.arange(self.robots.count, dtype=np.int32), self.device)
        self.num_envs = len(self.robot_indices)

        print("num_envs", self.num_envs)

        # initialize
        self.robots.initialize()
        self.robot_states = self.robots.get_world_poses()
        self.dof_pos = self.robots.get_joint_positions()

        self.initial_dof_pos = self.dof_pos
        self.dof_vel = self.robots.get_joint_velocities()
        self.initial_dof_vel = self.dof_vel

        self.xforms = XFormPrimView(self.xform_paths_expr)

    def calculate_grasp_location(self, keyword = "handle_", verticle = True, x_offset = 0.1):
        """
        Calculate the grasp location for the handle
        """
        bboxes_list = get_mesh_bboxes(keyword) 

        # assert len(bboxes_list) == self.num_envs, "more than one handle!"

        # get center and min x axis
        min_x = bboxes_list[0][0][0] # 
        center_list = [(e[1] + e[0]) / 2 for e in bboxes_list] # box center

        if verticle:
            grasp_list = [[min_x - x_offset, c[1], c[2] - 0.12] for c in center_list] 
        else:
            grasp_list = [[min_x - x_offset, c[1] + 0.12, c[2]] for c in center_list] 
 
        graps_pos = np.array(grasp_list, dtype=np.float32)
        
        base_rotation = [0.38268, 0, 0, 0.92388] if verticle else [0.3036, 0.23296, -0.56242, 0.73296]
        grasp_rot = np.array([base_rotation], dtype=np.float32)# XYZW
        
        # rotation: 0, 0.70711, 0, 0.70711; 0, 90, 0
        # rotation:[0.5, 0.5, 0.5, 0.5]

        return graps_pos, grasp_rot

    def move_to_target(self, goal_pos, goal_rot, finger = "thumb"):
        """
        Move hand to target points
        """
        # get end effector transforms
        finger_pos, finger_rot = self.xforms.get_world_poses()
        finger_rot = finger_rot[:,[1,2,3,0]] # WXYZ

        # get franka DOF states
        dof_pos = self.robots.get_joint_positions()

        # compute position and orientation error
        pos_err = goal_pos - finger_pos
        orn_err = orientation_error(goal_rot, finger_rot)
        dpose = np.concatenate([pos_err, orn_err], -1)[:, None].transpose(0, 2, 1)

        jacobians = self.robots._physics_view.get_jacobians()

        # jacobian entries corresponding to correct finger
        if finger == "thumb":
            finger_index = 14
        elif finger == "index":
            finger_index = 15
        elif finger == "middle":
            finger_index = 16
        elif finger == "pinky":
            finger_index = 17
        else: # ring
            finger_index = 18

        j_eef = jacobians[:, finger_index, :]

        # solve damped least squares
        j_eef_T = np.transpose(j_eef, (0, 2, 1))
        d = 0.05  # damping term
        lmbda = np.eye(6) * (d ** 2)
        u = (j_eef_T @ np.linalg.inv(j_eef @ j_eef_T + lmbda) @ dpose).reshape(self.num_envs, -1)

        # update position targets
        pos_targets = dof_pos + u  # * 0.3

        return pos_targets

    ##################################################################################################
    # -------------------------------------- Control ------------------------------------------------#
    ##################################################################################################

    def move_finger_to_fast(self, target_pos, target_rot, world, finger = "thumb", max_step = 100):
        """
        Quickly move the robot hands to the target position and rotation
        """
        for i in range(max_step):
            world.step(render=True)
    
            # get end effector transforms
            finger_pos, finger_rot = self.xforms.get_world_poses()
            finger_rot = finger_rot[:,[1,2,3,0]] # WXYZ -> XYZW

            print("finger_pos", finger_pos)
            
            orient_error = quat_mul(target_rot[0], quat_conjugate(finger_rot[0]))
            # print("orient_error", orient_error)
            # if abs(orient_error[3] - 1) < 0.02 and \
            #     np.sqrt(orient_error[0]**2 + orient_error[1]**2 + orient_error[2]**2) < 0.02 and \
            #     np.sqrt(np.sum((target_pos[0] - finger_pos[0])**2)) < 0.01:
            #     print("Done rotation, position", finger_pos, finger_rot)
            #     return 

            u = self.move_to_target(target_pos, target_rot)
            # u[:,[-2, -1]] = 0.05 if open_gripper else 0
            self.robots.set_joint_position_targets(u)
        
        print("Not Done rotation, position", finger_pos, finger_rot)

    def calculate_grasp_location_from_pred_box(self, box, verticle = True, x_offset = 0.1):
        """
        Calculate the grasp location for the handle
        """

        # assert len(bboxes_list) == self.num_envs, "more than one handle!"

        # get center and min x axis
        min_x = 0.618
        handle_y = 0.5 * (box[0] + box[2])
        handle_z = 0.5 * (box[1] + box[3])
        
        if verticle:
            grasp_list = [[min_x - x_offset, handle_y, handle_z - 0.12]] 
        else:
            grasp_list = [[min_x - x_offset, handle_y + 0.12, handle_z]] 
 
        graps_pos = np.array(grasp_list, dtype=np.float32)
        
        base_rotation = [0.38268, 0, 0, 0.92388] if verticle else [0.3036, 0.23296, -0.56242, 0.73296]
        grasp_rot = np.array([base_rotation], dtype=np.float32)# XYZW

        return graps_pos, grasp_rot