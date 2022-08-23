import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

from numpy_utils import *
from utils import get_mesh_bboxes

from omni.isaac.core import World, SimulationContext
from omni.isaac.core.prims.xform_prim_view import XFormPrimView
from omni.isaac.core.robots.robot_view import RobotView


default_grasp_profile = {
    "position_offset":{
        "vertical": [0,0,0],
        "horizontal": [0,0,0],
    },
    "rotation":{
        "vertical": [0,0,0,1], # XYZW
        "horizontal": [0,0,0,1],
    }
}

class HandBase():
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

        self.grasp_profile = default_grasp_profile

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

    def calculate_grasp_location(self, keyword = "handle_", verticle = True):
        """
        Calculate the grasp location for the handle
        """
        bboxes_list = get_mesh_bboxes(keyword) 

        # assert len(bboxes_list) == self.num_envs, "more than one handle!"

        # get center and min x axis
        min_x = bboxes_list[0][0][0] # 
        center_list = [(e[1] + e[0]) / 2 for e in bboxes_list] # box center

        pos_offset = self.grasp_profile["position_offset"]

        if verticle:
            v_pos_offset = pos_offset["vertical"]
            grasp_list = [[min_x - v_pos_offset[0], c[1] - v_pos_offset[1], c[2] - v_pos_offset[2]] for c in center_list] 
        else:
            h_pos_offset = pos_offset["horizontal"]
            grasp_list = [[min_x - h_pos_offset[0], c[1] - h_pos_offset[1], c[2] - h_pos_offset[2]] for c in center_list] 
 
        graps_pos = np.array(grasp_list, dtype=np.float32)
        
        base_rotation = self.grasp_profile["rotation"]["vertical"] if verticle else self.grasp_profile["rotation"]["horizontal"] 
        grasp_rot = np.array([base_rotation], dtype=np.float32)# XYZW
        
        # rotation: 0, 0.70711, 0, 0.70711; 0, 90, 0
        # rotation:[0.5, 0.5, 0.5, 0.5]

        return graps_pos, grasp_rot

    def calculate_grasp_location_from_pred_box(self, box, center = None, verticle = True):
        """
        Calculate the grasp location for the handle
        box: [y_0, z_0, y_1, z_1]
        center: [y, z]
        """

        # assert len(bboxes_list) == self.num_envs, "more than one handle!"

        # get center and min x axis
        min_x = 0.618
        handle_y = 0.5 * (box[0] + box[2])
        handle_z = 0.5 * (box[1] + box[3])

        if center:
            handle_y, handle_z = center
        
        pos_offset = self.grasp_profile["position_offset"]

        if verticle:
            v_pos_offset = pos_offset["vertical"]
            grasp_list = [[min_x - v_pos_offset[0], handle_y - v_pos_offset[1], handle_z - v_pos_offset[2]]] 
        else:
            h_pos_offset = pos_offset["horizontal"]
            grasp_list = [[min_x - h_pos_offset[0], handle_y - h_pos_offset[1], handle_z - h_pos_offset[2]]] 
 
        graps_pos = np.array(grasp_list, dtype=np.float32)
        
        base_rotation = self.grasp_profile["rotation"]["vertical"] if verticle else self.grasp_profile["rotation"]["horizontal"] 
        grasp_rot = np.array([base_rotation], dtype=np.float32)# XYZW

        return graps_pos, grasp_rot