import omni.ext
import omni.ui as ui

from .open_env import OpenEnv

# NOTE:
# go to directory: open-any-drawer/exts/open.any.drawer/open/any/drawer/
#  # start notebook from: /home/yizhou/.local/share/ov/pkg/isaac_sim-2022.1.0/jupyter_notebook.sh
#  start python: /home/yizhou/.local/share/ov/pkg/isaac_sim-2022.1.0/python.sh
# next paper about body language

# hand helper
import carb
import sys
from pxr import Usd, Sdf, PhysxSchema, UsdPhysics, Vt, Gf, UsdGeom, UsdShade
from omni.physx.scripts import physicsUtils, particleUtils
from omni.physx.scripts import deformableUtils, utils
import math
from copy import copy

from .hand.limiter import *

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class MyExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[open.any.drawer] MyExtension startup")

        self.env = OpenEnv()

        self._window = ui.Window("Open any drawer", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                with ui.HStack(height = 20):
                    ui.Button("Add Franka Robot", clicked_fn= self.env.add_robot)

                with ui.HStack(height = 20):
                    ui.Label("object index: ", width = 80)
                    self.object_id_ui = omni.ui.IntField(height=20, width = 40, style={ "margin": 2 })
                    self.object_id_ui.model.set_value(0)
                    ui.Label("object scale: ", width = 80)
                    self.object_scale_ui = omni.ui.FloatField(height=20, width = 40, style={ "margin": 2 })
                    self.object_scale_ui.model.set_value(0.1)
                    ui.Button("Add Object", clicked_fn=self.add_object)

                with ui.HStack(height = 20):
                    ui.Button("Add Ground", clicked_fn=self.add_ground)
                    ui.Button("Add Camera", clicked_fn=self.add_camera)

                with ui.HStack(height = 20):
                    # ui.Button("Add hand from copying", clicked_fn= self.debug)
                    ui.Button("Add hand from helper", clicked_fn= self.rig_hand2)
                    ui.Button("Rig D6", clicked_fn= self.debug_rig_d6)
                    ui.Button("Add drivers to joint", clicked_fn = self._add_driver_to_revolve_joint) 

                with ui.HStack(height = 20):
                    ui.Button("Test instructor", clicked_fn= self.debug_instructor)
                    ui.Button("Batch generation", clicked_fn= self.debug_batch_gen)
                
                with ui.HStack(height = 20):
                    ui.Button("Test task checker", clicked_fn= self.debug_task_checker)
                
                with ui.HStack(height = 20):
                    ui.Button("Test Load FastRCNN", clicked_fn= self.debug_load_model)
                

    def add_ground(self):
        from utils import add_ground_plane

        add_ground_plane("/World/Game")

    def add_camera(self):
        self.env.add_camera()
        self.env.setup_viewport()


    def add_object(self):
        object_id = self.object_id_ui.model.get_value_as_int()
        object_scale = self.object_scale_ui.model.get_value_as_float()
        self.env.add_object(object_id, scale = object_scale)

        selection = omni.usd.get_context().get_selection()
        selection.clear_selected_prim_paths()
        selection.set_prim_path_selected("/World/game", True, True, True, True)

        viewport = omni.kit.viewport_legacy.get_viewport_interface()
        if viewport:
            viewport.get_viewport_window().focus_on_selected()

    def on_shutdown(self):
        print("[open.any.drawer] MyExtension shutdown")


    
    def rig_hand2(self):
        print("debug2")
        from .hand.helper import HandHelper
        self.hand_helper = HandHelper()

    def debug_rig_d6(self):
        self._stage = omni.usd.get_context().get_stage()
        self._damping = 1e4
        self._stiffness = 2e5

        # create anchor:
        self._anchorXform = UsdGeom.Xform.Define(
            self._stage, Sdf.Path("/World/AnchorXform") # allegro/
        )
        # these are global coords because world is the xform's parent
        xformLocalToWorldTrans = Gf.Vec3f(0)
        xformLocalToWorldRot = Gf.Quatf(1.0)
        self._anchorXform.AddTranslateOp().Set(xformLocalToWorldTrans)
        self._anchorXform.AddOrientOp().Set(xformLocalToWorldRot)
      
        xformPrim = self._anchorXform.GetPrim()
        physicsAPI = UsdPhysics.RigidBodyAPI.Apply(xformPrim)
        physicsAPI.CreateRigidBodyEnabledAttr(True)
        physicsAPI.CreateKinematicEnabledAttr(True)

        # setup joint to floating hand base
        component = UsdPhysics.Joint.Define(
            self._stage, Sdf.Path("/World/AnchorToHandBaseD6") # allegro/
        )

        
        self._articulation_root = self._stage.GetPrimAtPath("/World/shadow_hand/robot0_hand_mount")  # "/World/Hand/Bones/l_carpal_mid" # "/World/allegro/allegro_mount"
        baseLocalToWorld = UsdGeom.Xformable(self._articulation_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        jointPosition = baseLocalToWorld.GetInverse().Transform(xformLocalToWorldTrans)
        jointPose = Gf.Quatf(baseLocalToWorld.GetInverse().RemoveScaleShear().ExtractRotationQuat())

        component.CreateExcludeFromArticulationAttr().Set(True)
        component.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
        component.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        component.CreateBody0Rel().SetTargets([self._anchorXform.GetPath()])

        component.CreateBody1Rel().SetTargets([self._articulation_root.GetPath()])
        component.CreateLocalPos1Attr().Set(jointPosition)
        component.CreateLocalRot1Attr().Set(jointPose)

        component.CreateBreakForceAttr().Set(sys.float_info.max)
        component.CreateBreakTorqueAttr().Set(sys.float_info.max)

        rootJointPrim = component.GetPrim()
        for dof in ["transX", "transY", "transZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, dof)
            driveAPI.CreateTypeAttr("force")
            # driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._damping)
            driveAPI.CreateStiffnessAttr(self._stiffness)

        for rotDof in ["rotX", "rotY", "rotZ"]:
            driveAPI = UsdPhysics.DriveAPI.Apply(rootJointPrim, rotDof)
            driveAPI.CreateTypeAttr("force")
            # driveAPI.CreateMaxForceAttr(self._drive_max_force)
            driveAPI.CreateTargetPositionAttr(0.0)
            driveAPI.CreateDampingAttr(self._damping)
            driveAPI.CreateStiffnessAttr(self._stiffness)

    def debug_instructor(self):
        print("debug instru")

        from task.instructor import SceneInstructor

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        self.scene_instr.build_handle_desc_ui()
        self.scene_instr.add_semantic_to_handle()

        self.scene_instr.export_data()

    def debug_batch_gen(self):
        print("debug_batch_gen")

        from .task.instructor import SceneInstructor
        import omni.replicator.core as rep

        # object_id = self.object_id_ui.model.set_value(4)
        object_id = self.object_id_ui.model.get_value_as_int()
        object_scale = self.object_scale_ui.model.get_value_as_float()
        self.env.add_object(object_id, scale = object_scale)

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        self.scene_instr.build_handle_desc_ui()
        
        print("scene_instr.is_obj_valid: ", self.scene_instr.is_obj_valid)
        if self.scene_instr.is_obj_valid:
            self.scene_instr.add_semantic_to_handle()
            self.scene_instr.output_path = f"/home/yizhou/Research/temp0/"
            self.scene_instr.export_data()
        
        
        # print("print(rep.orchestrator.get_is_started())", rep.orchestrator.get_is_started())
        
    ############ task check #####################################################################

    def debug_task_checker(self):
        print("debug_task_checker")
        from task.checker import TaskChecker
        from task.instructor import SceneInstructor

        self.env.add_robot()
        object_id = self.object_id_ui.model.get_value_as_int()
        object_scale = self.object_scale_ui.model.get_value_as_float()
        self.env.add_object(object_id, scale = object_scale)

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        self.scene_instr.build_handle_desc_ui()

        # self.task_checker = TaskChecker("mobility", "joint_0", "PhysicsRevoluteJoint")

    ############ deep learning #####################################################################

    def debug_load_model(self):
        print("load_model")
        from task.instructor import SceneInstructor

        self.scene_instr = SceneInstructor()
        self.scene_instr.analysis()
        self.scene_instr.build_handle_desc_ui()
        
        print("scene_instr.is_obj_valid: ", self.scene_instr.is_obj_valid)
        if self.scene_instr.is_obj_valid:
            # print("valid_handle_list",  self.scene_instr.valid_handle_list)
            self.scene_instr.load_model()
            self.scene_instr.predict_bounding_boxes(image_path="/home/yizhou/Research/temp0/rgb_0.png")
            print("pred bboxes", self.scene_instr.pred_boxes)

            handle_list = list(self.scene_instr.valid_handle_list.keys())
            for HANDLE_INDEX in range(len(handle_list)):
                handle_path_str = handle_list[HANDLE_INDEX]
                v_desc = self.scene_instr.valid_handle_list[handle_path_str]["vertical_description"]
                h_desc = self.scene_instr.valid_handle_list[handle_path_str]["horizontal_description"]
                
                print("handle_path_str", handle_path_str, "v desc: ", v_desc, "h desc:", h_desc)
                the_box = self.scene_instr.get_box_from_desc(v_desc, h_desc)
                print("the_box:", the_box)


            del self.scene_instr.model