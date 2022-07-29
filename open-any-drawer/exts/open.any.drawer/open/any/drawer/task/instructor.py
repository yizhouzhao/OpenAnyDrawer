# instructions as language

import omni
from pxr import UsdPhysics

from task.utils import *

class SceneInstructor():
    def __init__(self) -> None:
        # constant
        self.long_handle_ratio = 3 # ratio to determin the long handle
        self.short_handle_ratio = 1.5 # ratio to determin the small handle
        self.spatial_desc_tolerance = 0.05 # spatial description

        # scene
        self.stage = omni.usd.get_context().get_stage()

        # knowledge
        self.handle_knowledge = {}
        self.joint_knowledge = {"PhysicsRevoluteJoint":[], "PhysicsPrismaticJoint":[], "PhysicsFixedJoint": []}

    def analysis_handle(self):
        """
        Analysis handle to get the positions
        """

        keyword = "handle_"
        prim_list = list(self.stage.TraverseAll())
        prim_list = [ item for item in prim_list if keyword in item.GetPath().pathString and item.GetTypeName() == 'Mesh' ]

        # get basic information
        for prim in prim_list:
            prim_path_str = prim.GetPath().pathString

            # get bounding boxes
            bboxes = get_bounding_box(prim_path_str)
            center = 0.5 * (bboxes[0] + bboxes[1])
            scale = (bboxes[1][0] - bboxes[0][0], bboxes[1][1] - bboxes[0][1], bboxes[1][2] - bboxes[0][2])
            size = scale[0] * scale[1] * scale[2]

            size_type = self.get_handle_type_from_scale(scale)
            direction = "horizontal" if scale[1] > scale[2] else "vertical"

            self.handle_knowledge[prim_path_str] = {
                "center": center,
                "bboxes": bboxes,
                "scale": scale,
                "size": size,
                "size_type": size_type,
                "direction": direction,
                "overlap_with": [],
                "overlap_with_longer": False,

                "joint_type": "",
            }

        # get intersection
        for i in range(len(prim_list)):
            path_str1 = prim_list[i].GetPath().pathString
            bboxes1 = self.handle_knowledge[path_str1]["bboxes"]
            
            for j in range(i + 1, len(prim_list)):
                path_str2 = prim_list[j].GetPath().pathString
                bboxes2 = self.handle_knowledge[path_str2]["bboxes"]

                if bboxes_overlap(bboxes1, bboxes2):
                    overlap_with1 = self.handle_knowledge[path_str1]["overlap_with"]
                    overlap_with1.append(path_str2)

                    overlap_with2 = self.handle_knowledge[path_str2]["overlap_with"]
                    overlap_with2.append(path_str1)

                    if max(self.handle_knowledge[path_str1]["scale"]) > max(self.handle_knowledge[path_str2]["scale"]):
                        self.handle_knowledge[path_str2]["overlap_with_longer"] = True
                    else:
                        self.handle_knowledge[path_str1]["overlap_with_longer"] = True

    def analysis_cabinet_type(self):
        # get drawer/door from joint type
        stage = omni.usd.get_context().get_stage()
        prim_list = list(stage.TraverseAll())
        prim_list = [ item for item in prim_list if "joint_" in item.GetPath().pathString]

        # get joint knowledge
        for prim in prim_list:
            # print("type", prim, prim.GetTypeName())
            joint = UsdPhysics.Joint.Get(self.stage, prim.GetPath())
            assert joint, f"Not a joint? Check model {prim.GetPath().pathString}"
            b1paths = joint.GetBody1Rel().GetTargets()
            print("b1paths", prim.GetTypeName(), b1paths)
            self.joint_knowledge[prim.GetTypeName()].append(b1paths[0].pathString)

        # update joint type
        for handle_path_str in self.handle_knowledge:
            handle_know = self.handle_knowledge[handle_path_str]

            for joint_type in self.joint_knowledge:
                for joint_path_str in self.joint_knowledge[joint_type]:
                    if joint_path_str in handle_path_str:
                        handle_know["joint_type"] = joint_type
                        break
        
        # get revolute/linear handles
        self.revolute_handle_list = []
        self.linear_handle_list = []

        # if it doesn't overlap with any larger handle, it is a true handle
        for handle_path_str in self.handle_knowledge:
            if not self.handle_knowledge[handle_path_str]["overlap_with_longer"]:
                if self.handle_knowledge[handle_path_str]["joint_type"] == "PhysicsRevoluteJoint":
                    self.revolute_handle_list.append(handle_path_str)
                if self.handle_knowledge[handle_path_str]["joint_type"] == "PhysicsPrismaticJoint":
                    self.linear_handle_list.append(handle_path_str)

    def analysis_spatial_rel(self):
        """
        Analysis the spatial relationship of handle
        """
        print("knowledge", self.handle_knowledge)
                

    def get_handle_type_from_scale(self, scale):
        """
        Get a general shape for the handle
        """
        if max(scale) / min(scale) > self.long_handle_ratio:
            return "long"
        elif max(scale) / min(scale) < self.short_handle_ratio:
            return "short"
        else:
            return "middle?"



    