
import os
from typing import Union, Tuple, Dict, List, Callable

import omni.usd
import omni.kit.commands
from pxr import Sdf, UsdShade, Usd, Gf
import numpy as np


LOOKS_PATH = "/World/RLooks"

def material_omnipbr(
    prim_path_str,
    diffuse: Tuple[float] = None,
    diffuse_texture: str = None,
    roughness: float = None,
    roughness_texture: str = None,
    metallic: float = None,
    metallic_texture: str = None,
    specular: float = None,
    emissive_color: Tuple[float] = None,
    emissive_texture: str = None,
    emissive_intensity: float = 0.0,
    project_uvw: bool = False,
):
    stage = omni.usd.get_context().get_stage()
    mdl = "OmniPBR.mdl"
    mtl_name, _ = os.path.splitext(mdl)

    if not stage.GetPrimAtPath(LOOKS_PATH):
        stage.DefinePrim(LOOKS_PATH, "Scope")

    prim_path = omni.usd.get_stage_next_free_path(stage, f"{LOOKS_PATH}/{mdl.split('.')[0]}", False)
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim", mtl_url=mdl, mtl_name=mtl_name, mtl_path=prim_path, select_new_prim=False
    )
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(stage.GetPrimAtPath(prim_path), True))

    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float)
    shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float)
    shader.CreateInput("metallic_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("metallic_texture_influence", Sdf.ValueTypeNames.Float)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float)
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool)
    shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f)
    shader.CreateInput("emissive_color_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)

    enable_emission = emissive_intensity != 0.0
    roughness_texture_influence = float(roughness_texture is not None)
    metallic_texture_influence = float(roughness_texture is not None)


    prim = stage.GetPrimAtPath(prim_path)
    

    properties = {
        "diffuse_color_constant": diffuse,
        "diffuse_texture": diffuse_texture,
        "reflection_roughness_constant": roughness,
        "reflectionroughness_texture": roughness_texture,
        "reflection_roughness_texture_influence": roughness_texture_influence,
        "metallic_constant": metallic,
        "metallic_texture": metallic_texture,
        "metallic_texture_influence": metallic_texture_influence,
        "specular_level": specular,
        "enable_emission": enable_emission,
        "emissive_color": emissive_color,
        "emissive_color_texture": emissive_texture,
        "emissive_intensity": emissive_intensity,
        "project_uvw": project_uvw,
    }

    for attribute, attribute_value in properties.items():
        
        if attribute_value is None:
            continue

        if UsdShade.Material(prim):
            shader = UsdShade.Shader(omni.usd.get_shader_from_material(prim, True))
            shader.GetInput(attribute).Set(attribute_value)
        else:
            prim.GetAttribute(attribute).Set(attribute_value)


    omni.kit.commands.execute(
        "BindMaterialCommand", 
        prim_path=prim_path_str, 
        material_path=prim.GetPath().pathString, 
        strength=UsdShade.Tokens.strongerThanDescendants,
    )


def prim_random_color(prim_path_str):
    """
    Randomize color for prim at path
    """
    diffuse = Gf.Vec3f(np.random.rand(), np.random.rand(), np.random.rand())
    material_omnipbr(prim_path_str, diffuse = diffuse)


# # test
# prim_random_color("/World/Cube")
# print("test random shader")