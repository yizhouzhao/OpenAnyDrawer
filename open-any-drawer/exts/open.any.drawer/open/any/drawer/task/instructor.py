# instructions as language
import os
import carb
import omni
from pxr import UsdPhysics, Gf, UsdGeom

from task.utils import *

import omni.kit.viewport_widgets_manager as wm
from omni import ui


from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
import omni.replicator.core as rep

CAMERA_WIDGET_STYLING = {
    "Rectangle::background": {"background_color": 0x7F808080, "border_radius": 5}
}

class LabelWidget(wm.WidgetProvider):
    def __init__(self, text_list:list):
        self.text_list = text_list
    
    def build_widget(self, window):
        with ui.ZStack(width=0, height=0, style=CAMERA_WIDGET_STYLING):
            ui.Rectangle(name="background")
            with ui.VStack(width=0, height=0):
                ui.Spacer(height=2)
                for text in self.text_list:
                    ui.Label(text, width=0, height=0, name="", style={"color": "darkorange"})
                

class SceneInstructor():
    def __init__(self) -> None:
        # constant
        self.long_handle_ratio = 3 # ratio to determin the long handle
        self.short_handle_ratio = 1.5 # ratio to determin the small handle
        self.spatial_desc_tolerance = 0.05 # spatial description

        # output path
        self.output_path = "/home/yizhou/Research/temp"
        self.reset()

    def reset(self):
        # scene
        self.stage = omni.usd.get_context().get_stage()

        # knowledge
        self.handle_knowledge = {}
        self.joint_knowledge = {"PhysicsRevoluteJoint":[], "PhysicsPrismaticJoint":[], "PhysicsFixedJoint": []}

        # constant
        self.scale = 0.1 # object scale
        self.is_obj_valid = True # valid object scene

    ####################################################################################
    ############################ analysis ###############################################
    ####################################################################################

    def analysis(self):
        self.analysis_handle_primary()
        self.analysis_cabinet_type()
        self.analysis_spatial_rel()

    def analysis_handle_primary(self):
        """
        Analysis handle to get the positions
        """

        keyword = "handle_"
        prim_list = list(self.stage.TraverseAll())
        prim_list = [ item for item in prim_list if keyword in item.GetPath().pathString and item.GetTypeName() == 'Mesh' ]

        # get basic information
        for prim in prim_list:
            prim_path_str = prim.GetPath().pathString

            handle_num = prim_path_str.split("/")[-1].split("_")[-1]
            # get bounding boxes
            bboxes = get_bounding_box(prim_path_str)
            center = 0.5 * (bboxes[0] + bboxes[1])
            scale = (bboxes[1][0] - bboxes[0][0], bboxes[1][1] - bboxes[0][1], bboxes[1][2] - bboxes[0][2])
            size = scale[0] * scale[1] * scale[2]

            size_type = self.get_handle_type_from_scale(scale)
            direction = "horizontal" if scale[1] > scale[2] else "vertical"

            self.handle_knowledge[prim_path_str] = {
                "num": handle_num,
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
        self.valid_handle_list = {}

        # if it doesn't overlap with any larger handle, it is a true handle
        for handle_path_str in self.handle_knowledge:
            if not self.handle_knowledge[handle_path_str]["overlap_with_longer"]:
                if self.handle_knowledge[handle_path_str]["joint_type"] == "PhysicsRevoluteJoint":
                    self.valid_handle_list[handle_path_str] = {
                        "joint_type": "PhysicsRevoluteJoint",
                        "cabinet_type": "door",
                        "vertical_description": "",
                        "horizontal_description": "",
                    }
                if self.handle_knowledge[handle_path_str]["joint_type"] == "PhysicsPrismaticJoint":
                    self.valid_handle_list[handle_path_str] = {
                        "joint_type": "PhysicsPrismaticJoint",
                        "cabinet_type": "drawer",
                        "vertical_description": "",
                        "horizontal_description": "",
                    }
        

    def analysis_spatial_rel(self):
        """
        Analysis the spatial relationship of handle
        : joint_type  -> vertical -> horizontal
        """        
        print("analysis_spatial_rel: ", self.valid_handle_list)
        if len(self.valid_handle_list) == 0:
            carb.log_warn("No handle in the scene")
            self.is_obj_valid = False
            return
        
        # if only one joint, no need to describe from spatial layout
        if len(self.valid_handle_list) == 1:
            self.is_obj_valid = True
            return

        # get vertical and horizontal centers
        v_centers = []
        h_centers = []

        
        for handle_path_str in self.valid_handle_list:
            handle_center = self.handle_knowledge[handle_path_str]["center"]
            center_z = handle_center[2]
            center_y = handle_center[1]

            is_v_center_list = any([abs(z - center_z) < self.spatial_desc_tolerance for z in v_centers])
            is_h_center_list = any([abs(y - center_y) < self.spatial_desc_tolerance for y in h_centers])

            if not is_v_center_list:
                v_centers.append(center_z)
            if not is_h_center_list:
                h_centers.append(center_y)

        v_centers = sorted(v_centers)
        h_centers = sorted(h_centers)
        
        # vertical
        if len(v_centers) == 1:
            pass
        elif len(v_centers) == 2:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[2] - v_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "bottom"
                else:
                     self.valid_handle_list[handle_path_str]["vertical_description"] = "top"
        
        elif len(v_centers) == 3:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[2] - v_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "bottom"
                elif abs(handle_center[2] - v_centers[1]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "middle"
                else:
                     self.valid_handle_list[handle_path_str]["vertical_description"] = "top"
        
        elif len(v_centers) == 4:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[2] - v_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "bottom"
                elif abs(handle_center[2] - v_centers[1]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "second bottom"
                elif abs(handle_center[2] - v_centers[2]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "second top"
                else:
                     self.valid_handle_list[handle_path_str]["vertical_description"] = "top"
        else:
            carb.log_warn("too many handles align vertically!")
            self.is_obj_valid = False

        # horizontal
        if len(h_centers) == 1:
            pass
        elif len(h_centers) == 2:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[1] - h_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "right"
                else:
                     self.valid_handle_list[handle_path_str]["horizontal_description"] = "left"
        
        elif len(h_centers) == 3:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[1] - h_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "right"
                elif abs(handle_center[1] - h_centers[1]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "middle"
                else:
                     self.valid_handle_list[handle_path_str]["horizontal_description"] = "left"
        
        elif len(h_centers) == 4:
            for handle_path_str in self.valid_handle_list:
                handle_center = self.handle_knowledge[handle_path_str]["center"]
                if abs(handle_center[1] - h_centers[0]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "right"
                elif abs(handle_center[1] - h_centers[1]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "second right"
                elif abs(handle_center[1] - h_centers[2]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "second left"
                else:
                     self.valid_handle_list[handle_path_str]["horizontal_description"] = "left"
        else:
            carb.log_warn("too many handles align horizontally!")
            self.is_obj_valid = False

    
        # print("valid_handle_list: ", self.valid_handle_list)
        # print("knowledge", self.handle_knowledge)
                

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


    ####################################################################################
    ############################ UI ###############################################
    ####################################################################################

    def build_ui(self, desc:list, gui_path:str, gui_location):
        gui = self.stage.GetPrimAtPath(gui_path)
        if not gui:
            gui = UsdGeom.Xform.Define(self.stage, gui_path)
            gui.AddTranslateOp().Set(gui_location)
        self.wiget_id = wm.add_widget(gui_path, LabelWidget(desc), wm.WidgetAlignment.TOP)

    def build_handle_desc_ui(self):
        """
        build hud for handle
        """
        for handle_path_str in self.valid_handle_list:
            handle_center = self.handle_knowledge[handle_path_str]["center"]
            handle_num = self.handle_knowledge[handle_path_str]["num"]
            gui_location = handle_center
            gui_path = f"/World/GUI/handle_{handle_num}"

            h_desc = self.valid_handle_list[handle_path_str]["horizontal_description"]
            v_desc = self.valid_handle_list[handle_path_str]["vertical_description"]
            
            cabinet_type = self.valid_handle_list[handle_path_str]["cabinet_type"]

            self.build_ui([f"{cabinet_type}", "handle_" + handle_num, f"{v_desc}/{h_desc}"], gui_path, gui_location)

    ######################################## semantic #####################################################
    def add_semantic_to_handle(self):
        for handle_path_str in self.valid_handle_list:
            prim = self.stage.GetPrimAtPath(handle_path_str)
            add_update_semantics(prim, "handle")

    def export_data(self):
        """
        Export RGB and Bounding box info to file
        """
        with rep.new_layer():
            camera = rep.create.camera(position=(-10 * self.scale, 0, 5 * self.scale), rotation=(90, 0, -90))
            render_product = rep.create.render_product(camera, (256, 256))

             # Initialize and attach writer
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize( output_dir=self.output_path, rgb=True, bounding_box_2d_tight=True)
            writer.attach([render_product])

            with rep.trigger.on_frame(num_frames=1):
                pass

            rep.orchestrator.run()
            rep.BackendDispatch.wait_until_done()
            # rep.orchestrator.preview()
            # omni.kit.commands.execute("DeletePrims", paths=["/World/Game"])

