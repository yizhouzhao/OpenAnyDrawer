# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# ~/.local/share/ov/pkg/isaac_sim-2022.1.0/python.sh

"""Generate offline synthetic dataset
"""
from omni.isaac.kit import SimulationApp
import os

import carb

# Set rendering parameters and create an instance of kit
CONFIG = {"renderer": "RayTracedLighting", "headless": True, "width": 256, "height": 256, "num_frames": 10}

kit = SimulationApp(launch_config=CONFIG)

from omni.isaac.core import World

world = World()

from omni.isaac.core.prims.xform_prim import XFormPrim
from open_env import OpenEnv
from task.instructor import SceneInstructor

env = OpenEnv()


# we will be using the replicator library
import omni.replicator.core as rep

# This allows us to run replicator, which will update the random
# parameters and save out the data for as many frames as listed
def run_orchestrator():
    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        kit.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        kit.update()

    rep.BackendDispatch.wait_until_done()


for i in range(10, 200):
    env.add_object(i, scale = 0.1)
    game_obj = XFormPrim("/World/Game")
    game_obj_name = game_obj.name
    world.scene.add(game_obj)

    scene_instr = SceneInstructor()
    scene_instr.analysis()
    scene_instr.add_semantic_to_handle()

    if scene_instr.is_obj_valid:
        with rep.new_layer():
            camera = rep.create.camera(position=(-10 * scene_instr.scale, 0, 5 * scene_instr.scale), rotation=(90, 0, -90))
            render_product = rep.create.render_product(camera, (256, 256))

            # Initialize and attach writer
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize( output_dir=os.path.join(scene_instr.output_path, f"{i}"), rgb=True, bounding_box_2d_tight=True)
            writer.attach([render_product])

            with rep.trigger.on_frame(num_frames=1):
                pass
        
    
            run_orchestrator()

    world.scene.remove_object(game_obj_name)

kit.close()
