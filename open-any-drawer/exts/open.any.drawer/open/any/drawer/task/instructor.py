# instructions as language

import carb
import omni

import os
import torch
# try:
#     import cv2
# except:
#     omni.kit.pipapi.install("opencv-python")
#     import cv2
import numpy as np


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

        # pred
        self.pred_boxes = None
        self.is_pred_valid = True # Prediction valid

    ####################################################################################
    ############################ analysis ###############################################
    ####################################################################################

    def analysis(self):
        self.analysis_game()
        self.analysis_handle_primary()
        self.analysis_cabinet_type()
        self.analysis_spatial_rel()
    
    def analysis_game(self):
        """
        Analysis global game information
        """
        bboxes = get_bounding_box("/World/Game/mobility")
        self.game_center = 0.5 * (bboxes[0] + bboxes[1])

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
            relative_to_game_center = "left" if center[1] >= self.game_center[1] else "right"

            self.handle_knowledge[prim_path_str] = {
                "num": handle_num,
                "center": center,
                "relative_to_game_center": relative_to_game_center,
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
            # print("b1paths", prim.GetTypeName(), b1paths)
            self.joint_knowledge[prim.GetTypeName()].append([b1paths[0].pathString, prim.GetPath().pathString])

        # update joint type
        for handle_path_str in self.handle_knowledge:
            handle_know = self.handle_knowledge[handle_path_str]

            for joint_type in self.joint_knowledge:
                for joint_body_path_str, joint_prim_path_str in self.joint_knowledge[joint_type]:
                    if joint_body_path_str in handle_path_str:
                        handle_know["joint_type"] = joint_type
                        handle_know["joint_path_str"] = joint_prim_path_str
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
                
                # other import information
                self.valid_handle_list[handle_path_str]["joint"] = self.handle_knowledge[handle_path_str]["joint_path_str"].split("/")[-1]
                self.valid_handle_list[handle_path_str]["relative_to_game_center"]  = self.handle_knowledge[handle_path_str]["relative_to_game_center"]
                self.valid_handle_list[handle_path_str]["direction"]  = self.handle_knowledge[handle_path_str]["direction"]
                


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
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "second-bottom"
                elif abs(handle_center[2] - v_centers[2]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["vertical_description"] = "second-top"
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
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "second-right"
                elif abs(handle_center[1] - h_centers[2]) < self.spatial_desc_tolerance:
                    self.valid_handle_list[handle_path_str]["horizontal_description"] = "second-left"
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
            h_desc = self.valid_handle_list[handle_path_str]["horizontal_description"]
            v_desc = self.valid_handle_list[handle_path_str]["vertical_description"]
            
            cabinet_type = self.valid_handle_list[handle_path_str]["cabinet_type"]
            # add_update_semantics(prim, "handle")

            add_update_semantics(prim, semantic_label = f"{v_desc}_{h_desc}_{cabinet_type}")
            

    def export_data(self):
        """
        Export RGB and Bounding box info to file
        """
        with rep.new_layer():
            camera = rep.create.camera(position=(-10 * self.scale, 0, 5 * self.scale), rotation=(90, 0, -90))
            render_product = rep.create.render_product(camera, (256, 256))

             # Initialize and attach writer
            self.writer = rep.WriterRegistry.get("BasicWriter")
            self.writer.initialize( output_dir=self.output_path, rgb=True, bounding_box_2d_tight=True)
            self.writer.attach([render_product])

            with rep.trigger.on_frame(num_frames=1):
                pass

            rep.orchestrator.run()
            rep.BackendDispatch.wait_until_done()
            # rep.orchestrator.preview()
            # omni.kit.commands.execute("DeletePrims", paths=["/World/Game"])

    def load_model(self):
        """
        Load deep leanring model
        """
        from exp.model import load_vision_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = load_vision_model().to(self.device)
        print("successfully loaded model")
    
    def predict_bounding_boxes(self, image, detection_threshold = 0.5):
        """
        Predict bounding boxes
        ::params:
            image: 255 rgb
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = self.model.to(self.device)

        image_arr = image / 255.0

        images = [torch.tensor(image_arr).to(torch.float).permute(2,0,1).to(self.device )] # .to("cuda") 
        
        outputs = self.model(images)
        # print("outputs", outputs)
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()

        # sort from max to min
        inds = scores.argsort()[::-1]
        boxes = boxes[inds]

        # if no boxes!
        if len(boxes) == 0:
            self.is_pred_valid = False
            return


        select_boxes = boxes[scores >= detection_threshold].astype(np.int32)

        # if no boxes?
        if len(select_boxes) == 0:
            select_boxes = boxes
        
        # get world box positions
        self.pred_boxes= [self.get_bbox_world_position(box) for box in select_boxes]

    def get_bbox_world_position(self, box, 
           resolution = 256, D = -293, camera_pos = [-1, 0, 0.5], handle_x = 0.61857):
        """
        Calculate the grasp location for the handle

        box: [x_min, y_min, x_max, y_max] 2D boudning box in camera
        resolution: camera resolution
        D: depth of field
        camera_pos: camera_position
        handle_x: object offset

        """
        w_min = box[0] - resolution / 2
        w_max = box[2] - resolution / 2
        h_min = box[1] - resolution / 2
        h_max = box[3] - resolution / 2

        y_max = (handle_x - camera_pos[0]) * w_min / D + camera_pos[1]
        y_min = (handle_x - camera_pos[0]) * w_max / D + camera_pos[1]

        z_max = (handle_x - camera_pos[0]) * h_min / D + camera_pos[2]
        z_min = (handle_x - camera_pos[0]) * h_max / D + camera_pos[2]


        return [y_min, z_min, y_max, z_max]

    def get_box_from_desc(self, v_desc, h_desc):
        """
        Get box from description
        """

        # if no description, get bbox of the highest score
        if v_desc  == "" and h_desc == "":
            return self.pred_boxes[0]

        # if just one box
        if len(self.pred_boxes) == 1:
            return self.pred_boxes[0]
        
        v_boxes = sorted(self.pred_boxes, key = lambda box: 0.5 * (box[1] +  box[3]))
        h_boxes = sorted(self.pred_boxes, key = lambda box: 0.5 * (box[0] +  box[2]))
        # only vertical relation
        if h_desc == "":
            
            if v_desc == "top":
                return v_boxes[-1]
            elif v_desc == "second top" or v_desc == "middle":
                return v_boxes[-2]
            if v_desc == "bottom":
                return v_boxes[0]
            elif v_desc == "second bottom" or v_desc == "middle":
                return v_boxes[1]

        # only horizontal relation
        elif v_desc == "":
            
            if h_desc == "left":
                return h_boxes[-1]
            elif h_desc == "second left" or h_desc == "middle":
                return h_boxes[-2]
  

            if h_desc == "right":
                return h_boxes[0]
            elif h_desc == "second right" or h_desc == "middle":
                return h_boxes[1]

            
        else: # have both description
            if v_desc == "bottom" and h_desc == "left":
                if v_boxes[0][0] > v_boxes[1][0]:
                    return v_boxes[0]
                else:
                    return v_boxes[1]
            elif v_desc == "bottom" and h_desc == "right":
                if v_boxes[0][0] > v_boxes[1][0]:
                    return v_boxes[1]
                else:
                    return v_boxes[0]
            elif v_desc == "top" and h_desc == "left":
                if v_boxes[-1][0] > v_boxes[-2][0]:
                    return v_boxes[-1]
                else:
                    return v_boxes[-2]
            elif v_desc == "top" and h_desc == "right":
                if v_boxes[-1][0] > v_boxes[-2][0]:
                    return v_boxes[-2]
                else:
                    return v_boxes[-1]
            
            # TODO: unhandled situation
            else:
                return self.pred_boxes[0] 

        return self.pred_boxes[0]
    