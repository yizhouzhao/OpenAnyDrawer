import numpy as np
from PIL import Image

OBJ_INDEX = 4
SUCESS_PERCENTAGE = 0.15
result_file_path = "/home/yizhou/Research/Data/franka_exp.txt"

import getpass
user = getpass.getuser()


from omni.isaac.kit import SimulationApp

# "/home/yizhou/Research/OpenAnyDrawer/scene0.usd" # 
usd_path = f"omniverse://localhost/Users/{user}/scene0.usd"
simulation_app = SimulationApp({"headless": True, "open_usd": usd_path,  "livesync_usd": usd_path}) 

# world
from omni.isaac.core import World
world = World()

# custom import
from open_env import OpenEnv
from task.checker import TaskChecker
from task.instructor import SceneInstructor

env = OpenEnv()
env.setup_viewport()

# controller
from franka.control import FrankaControl
controller = FrankaControl("/World/Game*/Franka", "/World/Game*/Franka/panda_hand")
world.reset()
controller.start()
world.scene.add(controller.robots)

# show image
env.get_image().show()

# Add object
# TODO: iterate obj index
env.add_object(OBJ_INDEX, scale = 0.1)

from omni.isaac.core.prims.xform_prim import XFormPrim
mobility_obj = XFormPrim("/World/Game/mobility")
mobility_obj_name = mobility_obj.name

world.scene.add(mobility_obj)
world.reset()
world.render()

scene_instr = SceneInstructor()
scene_instr.analysis()
# scene_instr.build_handle_desc_ui()

# if not valid
if not scene_instr.is_obj_valid:
    print("object not valid: ", OBJ_INDEX)
    simulation_app.close()
    exit()

# TODO: iterate handle index
HANDLE_INDEX = 2

handle_path_str = list(scene_instr.valid_handle_list.keys())[HANDLE_INDEX]
handle_joint_type = scene_instr.handle_knowledge[handle_path_str]["joint_type"]
handle_joint = scene_instr.handle_knowledge[handle_path_str]["joint_path_str"].split("/")[-1]
rel_direction = scene_instr.handle_knowledge[handle_path_str]["relative_to_game_center"]

# Task
print("handle_path_str, handle_joint_type, handle_joint, rel_direction", handle_path_str, handle_joint_type, handle_joint, rel_direction)
task_checker = TaskChecker("mobility", handle_joint, handle_joint_type, IS_RUNTIME=True)

################################################## SOLUTION ##############################

# init
world.reset()

# FIXME: get grasp location vertical or horizontal
goal_pos, goal_rot = controller.calculate_grasp_location(keyword = handle_path_str, verticle = True)

# move above
goal_pos[...,0] -= 0.15
controller.move_hand_to_fast(goal_pos, goal_rot, world, max_step=300)

# move to
goal_pos[...,0] += 0.15
controller.move_hand_to_slow(goal_pos, goal_rot, world, step = 90)

# close gripper
for i in range(60):
    world.step(render=True)

    u = controller.move_to_target(goal_pos, goal_rot)
    u[:,[-2, -1]] = 0.5 - (0.5 - -1) / 60 * i
    controller.robots.set_joint_position_targets(u)

if handle_joint_type == "PhysicsRevoluteJoint": # door
    if rel_direction == "right":
        # 1. Open closewise

        # rotate and pull
        for i in range(0, 60):
            world.step(render=True)

            target_pos, target_rot = controller.calculate_pull_location(goal_pos, goal_rot, np.pi / 180 * i, 0.3, clock_wise=True)
            u = controller.move_to_target(target_pos, target_rot)
            u[:,[-2, -1]] = -1
            controller.robots.set_joint_position_targets(u)

    else:
        # 2. Open counter-closewise

        # rotate and pull
        for i in range(0, 60):
            world.step(render=True)

            target_pos, target_rot = controller.calculate_pull_location(goal_pos, goal_rot, np.pi / 180 * i, 0.3, 
                                                                        clock_wise=False)
            u = controller.move_to_target(target_pos, target_rot)
            u[:,[-2, -1]] = -1
            controller.robots.set_joint_position_targets(u)
            
else: # drawer
    # pull only
    for i in range(60):
        world.step(render=True)

        goal_pos[...,0] -= 2e-4 * i
        u = controller.move_to_target(goal_pos, goal_rot)
        u[:,[-2, -1]] = -1
        controller.robots.set_joint_position_targets(u)



# check task sucess
open_ratio = task_checker.joint_checker.compute_percentage()
task_success = open_ratio > SUCESS_PERCENTAGE
print("open_ratio, task_success", open_ratio, task_success)

with open(result_file_path, "a") as f:
    f.write(f"{OBJ_INDEX}\t{HANDLE_INDEX}\t{handle_joint_type}\t{task_success}")

env.get_image().show()

# reset
world.reset()

# close object
world.scene.remove_object(mobility_obj_name)
world.render()


simulation_app.close()
