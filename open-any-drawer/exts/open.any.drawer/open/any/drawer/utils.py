# utility

import omni
from omni.physx.scripts import physicsUtils
from pxr import UsdGeom, Usd, UsdShade, UsdPhysics, Gf

def add_ground_plane(prim_path = "/World/game", invisible = False):
    stage = omni.usd.get_context().get_stage()
    purposes = [UsdGeom.Tokens.default_]
    bboxcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    prim = stage.GetPrimAtPath(prim_path)
    bboxes = bboxcache.ComputeWorldBound(prim)
    # print("bboxes", bboxes)
    z = bboxes.ComputeAlignedRange().GetMin()[2]
    physicsUtils.add_ground_plane(stage, "/groundPlane", "Z", 10.0, Gf.Vec3f(0.0, 0.0, z), Gf.Vec3f(0.2))
    
    if invisible:
        prim_list = list(stage.TraverseAll())
        prim_list = [ item for item in prim_list if 'groundPlane' in item.GetPath().pathString and item.GetTypeName() == 'Mesh' ]
        for prim in prim_list:
            prim.GetAttribute('visibility').Set('invisible')

def get_bounding_box(prim_path: str):
    """
    Get the bounding box of a prim
    """
    stage = omni.usd.get_context().get_stage()

    purposes = [UsdGeom.Tokens.default_]
    bboxcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    prim = stage.GetPrimAtPath(prim_path)
    bboxes = bboxcache.ComputeWorldBound(prim)
    # print("bboxes", bboxes)
    game_bboxes = [bboxes.ComputeAlignedRange().GetMin(),bboxes.ComputeAlignedRange().GetMax()]
    
    return game_bboxes

def add_physical_material_to(keyword:str):
    """
    Set up physical material
    """
    stage = omni.usd.get_context().get_stage()
    prim_list = list(stage.TraverseAll())
    prim_list = [ item for item in prim_list if keyword in item.GetPath().pathString and item.GetTypeName() == 'Mesh' ]
    for prim in prim_list:
        setup_physics_material(prim)
        print("add physics material to handle")
        # setStaticCollider(prim, approximationShape = "convexDecomposition")

def setup_physics_material(prim):
    """
    Set up physic material for prim at Path
    """
    # def _setup_physics_material(self, path: Sdf.Path):
    stage = omni.usd.get_context().get_stage()
    _material_static_friction = 100.0
    _material_dynamic_friction = 100.0
    _material_restitution = 0.0
    _physicsMaterialPath = None

    if _physicsMaterialPath is None:
        # _physicsMaterialPath = stage.GetDefaultPrim().GetPath().AppendChild("physicsMaterial")
        _physicsMaterialPath = prim.GetPath().AppendChild("physicsMaterial")
        
        UsdShade.Material.Define(stage, _physicsMaterialPath)
        material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(_physicsMaterialPath))
        material.CreateStaticFrictionAttr().Set(_material_static_friction)
        material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
        material.CreateRestitutionAttr().Set(_material_restitution)

    collisionAPI = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
    # prim = stage.GetPrimAtPath(path)
    if not collisionAPI:
        collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
    # apply material
    physicsUtils.add_physics_material_to_prim(stage, prim, _physicsMaterialPath)
    print("physics material: path: ", _physicsMaterialPath)

def fix_linear_joint(fix_driver = True, damping_cofficient = 0.0):
    stage = omni.usd.get_context().get_stage()
    prim_list = stage.TraverseAll()
    for prim in prim_list:
        if "joint_" in str(prim.GetPath()):
            if fix_driver:
                # find linear drive
                joint_driver = UsdPhysics.DriveAPI.Get(prim, "linear")
                if joint_driver:
                    joint_driver.CreateDampingAttr(damping_cofficient)

                # find linear drive
                joint_driver = UsdPhysics.DriveAPI.Get(prim, "angular")
                if joint_driver:
                    joint_driver.CreateDampingAttr(damping_cofficient)
            
            # find linear joint upperlimit
            joint = UsdPhysics.PrismaticJoint.Get(stage, prim.GetPath())	
            if joint:
                upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
                # print(prim.GetPath(), "upper_limit", upper_limit)
                mobility_prim = prim.GetParent().GetParent()
                mobility_xform = UsdGeom.Xformable.Get(stage, mobility_prim.GetPath())
                scale_factor = mobility_xform.GetOrderedXformOps()[2].Get()[0]
                # print("scale_factor", scale_factor)
                joint.CreateUpperLimitAttr(upper_limit * scale_factor / 100)