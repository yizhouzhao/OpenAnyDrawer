import numpy as np
from PIL import Image

## Get object indexes
# import os
# OBJ_INDEX_LIST = []
# for i in os.listdir("/home/yizhou/Research/temp/"):
#     if str(i).isdigit():
#         OBJ_INDEX_LIST.append(i)
# print(sorted(OBJ_INDEX_LIST, key = lambda x: int(x)))

OBJ_INDEX_LIST = ['0', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '30', '31', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '52', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '105', '106', '107', '108', '110', '111', '112', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '134', '135', '136', '137', '138', '139', '140', '142', '143', '144', '145', '146', '147', '148', '149', '151', '152', '153', '154', '156', '157', '158', '159', '160', '162', '163', '164', '165', '168', '169', '170', '171', '172', '173', '175', '176', '177', '179', '180', '182', '183', '184', '185', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197']

SUCESS_PERCENTAGE = 20
result_file_path = "/home/yizhou/Research/Data/franka_exp.txt"
SHOW_IMAGE = False

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
from omni.isaac.core.prims.xform_prim import XFormPrim

env = OpenEnv()
env.setup_viewport()

# controller
from franka.control import FrankaControl
controller = FrankaControl("/World/Game*/Franka", "/World/Game*/Franka/panda_hand")
world.reset()
controller.start()
world.scene.add(controller.robots)

# show image
if SHOW_IMAGE:
    env.get_image().show()

# Add object
# TODO: iterate obj index
for OBJ_INDEX in OBJ_INDEX_LIST[:20]:
    env.add_object(OBJ_INDEX, scale = 0.1)
    
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
    handle_num = len(list(scene_instr.valid_handle_list.keys()))

    for HANDLE_INDEX in range(handle_num):
        handle_path_str = list(scene_instr.valid_handle_list.keys())[HANDLE_INDEX]
        handle_joint_type = scene_instr.valid_handle_list[handle_path_str]["joint_type"]
        handle_joint = scene_instr.valid_handle_list[handle_path_str]["joint"]
        handle_rel_direciton = scene_instr.valid_handle_list[handle_path_str]["relative_to_game_center"]
        handle_direction = scene_instr.valid_handle_list[handle_path_str]["direction"]

        # Task
        print("handle_path_str, handle_joint_type, handle_joint, rel_direction", handle_path_str, handle_joint_type, handle_joint, handle_rel_direciton)
        task_checker = TaskChecker("mobility", handle_joint, handle_joint_type, IS_RUNTIME=True)

        ################################################## SOLUTION ##############################

        # init
        world.reset()

        # FIXME: get grasp location vertical or horizontal
        goal_pos, goal_rot = controller.calculate_grasp_location(keyword = handle_path_str, verticle = handle_direction == "horizontal")

        # move above
        goal_pos[...,0] -= 0.1
        controller.move_hand_to_fast(goal_pos, goal_rot, world, max_step=1200)

        # move to
        goal_pos[...,0] += 0.1
        controller.move_hand_to_slow(goal_pos, goal_rot, world, step = 90)

        # close gripper
        for i in range(60):
            world.step(render=True)

            u = controller.move_to_target(goal_pos, goal_rot)
            u[:,[-2, -1]] = 0.5 - (0.5 - -1) / 60 * i
            controller.robots.set_joint_position_targets(u)

        if handle_joint_type == "PhysicsRevoluteJoint": # door
            if handle_rel_direciton == "right":
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

        # wait some time
        for i in range(0, 60):
            world.step(render=True)

        # check task sucess
        open_ratio = task_checker.joint_checker.compute_percentage()
        if handle_joint_type == "PhysicsRevoluteJoint": # open a door the upper limit may reach 180 degree
            open_ratio *= 2
        task_success = open_ratio > SUCESS_PERCENTAGE 
        print("open_ratio, task_success", open_ratio, task_success)

        with open(result_file_path, "a") as f:
            f.write(f"{OBJ_INDEX},{HANDLE_INDEX},{handle_path_str},{handle_joint_type},{handle_joint},{task_success},{open_ratio}\n")

        if SHOW_IMAGE:
            world.render()
            env.get_image().show()

        # reset
        world.reset()

    # close object
    world.scene.remove_object(mobility_obj_name)
    world.render()

simulation_app.close()
