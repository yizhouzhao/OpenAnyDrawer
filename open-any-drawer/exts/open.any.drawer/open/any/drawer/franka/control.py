import os
import sys
sys.path.append(os.path.dirname(__file__))

import omni
import pxr
from pxr import Gf

from omni.isaac.franka import Franka
from omni.isaac.core.utils.stage import set_stage_up_axis

from omni.isaac.core import World, SimulationContext
from omni.isaac.synthetic_utils import SyntheticDataHelper

from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.robots.robot_view import RobotView

import numpy as np
from pathlib import Path


from numpy_utils import *
from utils import get_mesh_bboxes

ROOT = str(Path(__file__).parent.joinpath("../../../../../../").resolve())

class FrankaControl():
    
    def __init__(self,  
        prim_paths_expr="",
        xform_paths_expr="",
        backend = "numpy",
        device = None) -> None: 
        
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

    def move_to_target(self, goal_pos, goal_rot):
        """
        Move hand to target points
        """
        # get end effector transforms
        hand_pos, hand_rot = self.xforms.get_world_poses()
        hand_rot = hand_rot[:,[1,2,3,0]] # WXYZ

        # get franka DOF states
        dof_pos = self.robots.get_joint_positions()

        # compute position and orientation error
        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(goal_rot, hand_rot)
        dpose = np.concatenate([pos_err, orn_err], -1)[:, None].transpose(0, 2, 1)

        jacobians = self.robots._physics_view.get_jacobians()

        # jacobian entries corresponding to franka hand
        franka_hand_index = 8  # !!!
        j_eef = jacobians[:, franka_hand_index - 1, :]

        # solve damped least squares
        j_eef_T = np.transpose(j_eef, (0, 2, 1))
        d = 0.05  # damping term
        lmbda = np.eye(6) * (d ** 2)
        u = (j_eef_T @ np.linalg.inv(j_eef @ j_eef_T + lmbda) @ dpose).reshape(self.num_envs, 9)

        # update position targets
        pos_targets = dof_pos + u  # * 0.3

        return pos_targets

    ##################################################################################################
    # -------------------------------------- Calculation --------------------------------------------#
    ##################################################################################################

    def calculate_grasp_location(self, keyword = "handle_", verticle = True, x_offset = 0.086):
        """
        Calculate the grasp location for the handle
        """
        bboxes_list = get_mesh_bboxes(keyword) 

        assert len(bboxes_list) == self.num_envs, "more than one handle!"

        # get center and min x axis
        min_x = bboxes_list[0][0][0] # 
        center_list = [(e[1] + e[0]) / 2 for e in bboxes_list] # box center

        grasp_list = [[min_x - x_offset, c[1], c[2]] for c in center_list]

        graps_pos = np.array(grasp_list, dtype=np.float32)
        
        base_rotation = [0.5, 0.5, 0.5, 0.5] if verticle else [0, 0.70711, 0, 0.70711]
        grasp_rot = np.array([base_rotation], dtype=np.float32).repeat(self.num_envs, axis = 0) # XYZW
        
        # rotation: 0, 0.70711, 0, 0.70711; 0, 90, 0
        # rotation:[0.5, 0.5, 0.5, 0.5]

        return graps_pos, grasp_rot

    def calculate_pull_location(self, start_pos, start_rot, theta, r, clock_wise = False):
        """
        Calculate how to pull to open the Cabinet
        """
        clock_wise = float(2 * clock_wise - 1)
        
        # position
        pos_offset = np.tile(np.array([-r * np.sin(theta),  clock_wise * r * (1 - np.cos(theta)), 0]), (self.num_envs, 1))
        target_pos = start_pos + pos_offset

        # rotate
        rot_offset = np.tile(np.array([np.sin(clock_wise * theta / 2), 0, 0, np.cos( - clock_wise * theta / 2)]), (self.num_envs, 1))
        target_rot = quat_mul(start_rot, rot_offset)

        return target_pos, target_rot


    ##################################################################################################
    # -------------------------------------- Control ------------------------------------------------#
    ##################################################################################################

    def move_hand_to_fast(self, target_pos, target_rot, world, open_gripper = True, max_step = 300):
        """
        Quickly move the robot hands to the target position and rotation
        """
        for i in range(max_step):
            world.step(render=True)
    
            # get end effector transforms
            hand_pos, hand_rot = self.xforms.get_world_poses()
            hand_rot = hand_rot[:,[1,2,3,0]] # WXYZ -> XYZW
            
            orient_error = quat_mul(target_rot[0], quat_conjugate(hand_rot[0]))
            # print("orient_error", orient_error)
            if abs(orient_error[3] - 1) < 0.02 and \
                np.sqrt(orient_error[0]**2 + orient_error[1]**2 + orient_error[2]**2) < 0.02 and \
                np.sqrt(np.sum((target_pos[0] - hand_pos[0])**2)) < 0.01:
                print("Done rotation, position", hand_pos, hand_rot)
                return 

            u = self.move_to_target(target_pos, target_rot)
            u[:,[-2, -1]] = 0.05 if open_gripper else 0
            self.robots.set_joint_position_targets(u)
        
        print("Not Done rotation, position", hand_pos, hand_rot)

    def move_hand_to_slow(self, target_pos, target_rot, world, open_gripper = True, step = 60):
        """
        Continuously and slowly move robot hands to the target position and rotation
        target_pos, target_rot: [x,y,z], [x, y, z, w]
        """
        hand_pos, hand_rot = self.xforms.get_world_poses() # [x,y,z], [w, x, y, z]
        hand_rot = hand_rot[:,[1,2,3,0]] # WXYZ -> XYZW

        inter_pos, inter_rot = np.zeros_like(hand_pos), np.zeros_like(hand_rot)
        start_pos, start_rot = [], []
        target_pos_gf, target_rot_gf = [], []
        
        # init
        for i in range(self.num_envs):
            start_pos.append(Gf.Vec3f(float(hand_pos[i][0]), float(hand_pos[i][1]), float(hand_pos[i][2])))
            start_rot.append(Gf.Quatf(float(hand_rot[i][3]),float(hand_rot[i][0]),float(hand_rot[i][1]),float(hand_rot[i][2])))

            target_pos_gf.append(Gf.Vec3f(float(target_pos[i][0]), float(target_pos[i][1]), float(target_pos[i][2])))
            target_rot_gf.append(Gf.Quatf(float(target_rot[i][3]),float(target_rot[i][0]),float(target_rot[i][1]),float(target_rot[i][2])))

        # gripper 
        dof_pos = self.robots.get_joint_positions()
        init_gripper_close = dof_pos[...,-1][0] <= 0.015

        # step
        for t in range(step):
            world.step(render=True)

            for i in range(self.num_envs):
                inter_pos_i = Gf.Lerp(t / (step - 1), start_pos[i], target_pos_gf[i])
                inter_pos[i] = [inter_pos_i[0], inter_pos_i[1], inter_pos_i[2]]

                inter_rot_i = Gf.Slerp(t / (step - 1), start_rot[i], target_rot_gf[i])
                inter_rot_i_imaginary = inter_rot_i.GetImaginary()
                inter_rot[i] = [inter_rot_i_imaginary[0], inter_rot_i_imaginary[1], inter_rot_i_imaginary[2], inter_rot_i.GetReal()]
            
            u = self.move_to_target(inter_pos, inter_rot)
            if init_gripper_close and not open_gripper:
                gripper_target = -0.5
            else:
                gripper_target = 0.5 if open_gripper else 0.5 - (0.5 - -0.5) / (step - 1) * t
            # print("gripper_target", gripper_target)
            u[:,[-2, -1]] = gripper_target
            self.robots.set_joint_position_targets(u)

        # final adjustment
        for t in range(step // 10):
            world.step(render=True)

            u = self.move_to_target(target_pos, target_rot)
            u[:,[-2, -1]] = 0.5 if open_gripper else -0.5
            self.robots.set_joint_position_targets(u)

        world.step(render=True)
    

    ############################### SLAM #########################################################
    

    def calculate_grasp_location_from_bbox(self, box, 
           resolution = 256, D = -293, camera_pos = [-1, 0, 0.5], handle_x = 0.61857):
        """
        Calculate the grasp location for the handle

        box: [x_min, y_min, x_max, y_max] 2D boudning box in camera
        resolution: camera resolution
        D: depth of field
        camera_pos: camera_position
        handle_x: object offset

        """
        delta_w = (box[0] + box[2]) / 2 - resolution / 2
        delta_h = (box[1] + box[3]) / 2 - resolution / 2

        handle_z = (handle_x - camera_pos[0]) * delta_h / D + camera_pos[2]
        handle_y = (handle_x - camera_pos[0]) * delta_w / D + camera_pos[1]

        graps_pos = np.array([[handle_x, handle_y, handle_z]], dtype=np.float32)
        verticle = delta_w < delta_h

        base_rotation = [0.5, 0.5, 0.5, 0.5] if verticle else [0, 0.70711, 0, 0.70711]
        grasp_rot = np.array([base_rotation], dtype=np.float32).repeat(self.num_envs, axis = 0) # XYZW
        
        # rotation: 0, 0.70711, 0, 0.70711; 0, 90, 0
        # rotation:[0.5, 0.5, 0.5, 0.5]

        return graps_pos, grasp_rot 



