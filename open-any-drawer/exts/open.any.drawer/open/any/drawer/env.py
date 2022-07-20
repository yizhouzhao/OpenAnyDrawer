import omni
import pxr

from omni.isaac.franka import Franka
from omni.isaac.core.utils.stage import set_stage_up_axis

import os
from pathlib import Path

ROOT = str(Path(__file__).parent.joinpath("../../../../../../").resolve())

class OpenEnv():
    
    def __init__(self) -> None:
        pass

    def add_robot(self):
        print("add robot")

        
        set_stage_up_axis("z")

        # import robot
        self.robot = Franka("/World/Franka")

    def add_object(self):
        from .utils import get_bounding_box, add_physical_material_to, fix_linear_joint

        print("add object")
        self.stage = omni.usd.get_context().get_stage()

        self.game_path_str = "/World/Game"
        xform_game = self.stage.GetPrimAtPath(self.game_path_str)
        if not xform_game:
            xform_game = pxr.UsdGeom.Xform.Define(self.stage, self.game_path_str)

        # move obj to the correct place
        mobility_prim_path = xform_game.GetPath().pathString + "/mobility"
        prim = self.stage.GetPrimAtPath(mobility_prim_path)
        if not prim.IsValid():
            prim = self.stage.DefinePrim(mobility_prim_path)
        
        object_ids = os.listdir(os.path.join(ROOT, "Asset/Sapien/StorageFurniture/"))
        obj_usd_path = os.path.join(ROOT, f"Asset/Sapien/StorageFurniture/{object_ids[0]}/mobility.usd")
        success_bool = prim.GetReferences().AddReference(obj_usd_path)
        assert success_bool, f"Import error at usd {obj_usd_path}"

        
        xform = pxr.Gf.Matrix4d().SetRotate(pxr.Gf.Quatf(1.0,0.0,0.0,0.0)) * \
            pxr.Gf.Matrix4d().SetTranslate([0,0,0]) * \
                pxr.Gf.Matrix4d().SetScale([0.7,0.7,0.7])
        
        omni.kit.commands.execute(
                "TransformPrimCommand", 
                path=mobility_prim_path,
                new_transform_matrix=xform,
            )

        # get obj bounding box
        bboxes = get_bounding_box(mobility_prim_path)
        position = [-bboxes[0][0] + 0.5, 0, -bboxes[0][2]]
        xform.SetTranslateOnly(position)

        omni.kit.commands.execute(
                "TransformPrimCommand", 
                path=mobility_prim_path,
                new_transform_matrix=xform,
            )

        # add physical meterial to
        add_physical_material_to("handle_")
        
        # fix linear joint
        fix_linear_joint()