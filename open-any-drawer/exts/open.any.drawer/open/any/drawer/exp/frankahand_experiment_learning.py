import numpy as np
from PIL import Image

## Get object indexes
# import os
# OBJ_INDEX_LIST = []
# for i in os.listdir("/home/yizhou/Research/temp/"):
#     if str(i).isdigit():
#         OBJ_INDEX_LIST.append(i)
# print(sorted(OBJ_INDEX_LIST, key = lambda x: int(x)))

from exp.params import OBJ_INDEX_LIST

SUCESS_PERCENTAGE = 20
result_file_path = "/home/yizhou/Research/Data/frankahand_exp_learning.txt"
MODEL_PATH = "/home/yizhou/Research/temp0/fasterrcnn_resnet50_fpn.pth"
SHOW_IMAGE = False


import getpass
user = getpass.getuser()


from omni.isaac.kit import SimulationApp

# "/home/yizhou/Research/OpenAnyDrawer/scene0.usd" # 
usd_path = f"omniverse://localhost/Users/{user}/scene4.usd"
simulation_app = SimulationApp({"headless": True, "open_usd": usd_path,  "livesync_usd": usd_path}) 

# world
import omni
from omni.isaac.core import World
world = World()

# reset scene
mobility_prim = world.scene.stage.GetPrimAtPath("/World/Game/mobility")
if mobility_prim:
    omni.kit.commands.execute("DeletePrims", paths=["/World/Game/mobility"])

# reset scene
replicator_prim = world.scene.stage.GetPrimAtPath("/Replicator")
if replicator_prim:
    omni.kit.commands.execute("DeletePrims", paths=["/Replicator"])


# custom import
from open_env import OpenEnv
from franka.gripper import GripperHandEnv
from task.checker import TaskChecker
from task.instructor import SceneInstructor
from omni.isaac.core.prims.xform_prim import XFormPrim

env = OpenEnv()
env.add_camera()
env.setup_viewport()

# env = HandEnv("/World/Hand/Bones/l_carpal_mid", "/World/Hand*/Bones/l_thumbSkeleton_grp/l_distalThumb_mid")
controller = GripperHandEnv("/World/Franka/panda_link8", "/World/AnchorXform")

# init
world.reset()
controller.start()
world.scene.add(controller.robots)

# hide robot
hand_prim = world.scene.stage.GetPrimAtPath("/World/Franka")
hand_prim.GetAttribute('visibility').Set('invisible')

if SHOW_IMAGE:
    world.render()
    env.get_image()

# load deep leanrning model
from exp.model import load_vision_model
model = load_vision_model(model_path = MODEL_PATH, model_name = "fasterrcnn_resnet50_fpn")

# iterate object index
for OBJ_INDEX in OBJ_INDEX_LIST[:1]:
    OBJ_INDEX = int(OBJ_INDEX)

    env.add_object(OBJ_INDEX, scale = 0.1)

    mobility_obj = XFormPrim("/World/Game/mobility")
    mobility_obj_name = mobility_obj.name

    world.scene.add(mobility_obj)
    world.reset()
    world.render()

    scene_instr = SceneInstructor()
    scene_instr.analysis()

    # export data and load model
    # scene_instr.output_path = "/home/yizhou/Research/temp0/"
    # scene_instr.export_data()
    # omni.kit.commands.execute("DeletePrims", paths=["/Replicator"])
    world.render()
    world.render()
    world.render()
    image_array =env.get_image(return_array=True)

    scene_instr.model = model
    scene_instr.predict_bounding_boxes(image_array[:,:,:3])

    # if not valid
    if not scene_instr.is_obj_valid:
        print("object not valid: ", OBJ_INDEX)
        simulation_app.close()
        exit()
    
    # if no valid predicted boundbox
    if not scene_instr.is_pred_valid:
        with open(result_file_path, "a") as f:
            f.write(f"{OBJ_INDEX}, invalid prediction\n")
        
        world.scene.remove_object(mobility_obj_name)
        world.reset()
        controller.xforms.set_world_poses(positions=np.array([[0,0,0]]), orientations = np.array([[1, 0, 0, 0]])) # WXYZ
        for _ in range(30):
            world.step()

        continue


    # iterate handle index
    handle_num = len(list(scene_instr.valid_handle_list.keys()))

    for HANDLE_INDEX in range(handle_num):
        handle_path_str = list(scene_instr.valid_handle_list.keys())[HANDLE_INDEX]
        handle_joint_type = scene_instr.valid_handle_list[handle_path_str]["joint_type"]
        handle_joint = scene_instr.valid_handle_list[handle_path_str]["joint"]
        handle_rel_direciton = scene_instr.valid_handle_list[handle_path_str]["relative_to_game_center"]
        # handle_direction = scene_instr.valid_handle_list[handle_path_str]["direction"]
        
        # Task
        print("handle_path_str, handle_joint_type, handle_joint, rel_direction", handle_path_str, handle_joint_type, handle_joint, handle_rel_direciton)
        task_checker = TaskChecker("mobility", handle_joint, handle_joint_type, IS_RUNTIME=True)

        ################################################## LEARNING SOLUTION ##############################

     
        v_desc = scene_instr.valid_handle_list[handle_path_str]["vertical_description"]
        h_desc = scene_instr.valid_handle_list[handle_path_str]["horizontal_description"]
             
        the_box = scene_instr.get_box_from_desc(v_desc, h_desc)
        handle_direction = "horizontal" if (the_box[2] - the_box[0]) > (the_box[3] - the_box[1]) else "vertical" 

        # init
        world.reset()
        controller.xforms.set_world_poses(positions=np.array([[0,0,0]]), orientations = np.array([[1, 0, 0, 0]])) # WXYZ
        for _ in range(60):
            world.step() # wait some time
        
        # get grasp location, if handle is horizontal, gripper should be vertical
        # graps_pos, grasp_rot = controller.calculate_grasp_location(keyword = handle_path_str, 
        #                                                        verticle = handle_direction == "horizontal")
        
        graps_pos, grasp_rot = controller.calculate_grasp_location_from_pred_box(the_box, verticle= handle_direction == "horizontal")
        print("graps_pos, grasp_rot ", graps_pos, grasp_rot )
        
        # move close to handle
        graps_pos[...,0] -= 0.1
        controller.xforms.set_world_poses(graps_pos, grasp_rot)
        for _ in range(200):
            world.step(render=False)         

        # move to handle
        graps_pos[...,0] += 0.1
        controller.xforms.set_world_poses(graps_pos, grasp_rot)
        for _ in range(100):
            world.step(render=False)        

        # close
        pos = np.array([[0.0, 0.0]])
                    
        for _ in range(100):
            pos -= 0.01
            controller.robots.set_joint_position_targets(pos)
            world.step(render=False)
        
        # pull out
        for i in range(300):
            graps_pos[...,0] -= 0.001
            controller.xforms.set_world_poses(graps_pos, grasp_rot)
            controller.robots.set_joint_position_targets(pos)
            pos += 0.015
            world.step(render=False)

        # check task sucess
        open_ratio = task_checker.joint_checker.compute_percentage()
        if handle_joint_type == "PhysicsRevoluteJoint": # open a door the upper limit may reach 180 degree
            open_ratio *= 2
        task_success = open_ratio > SUCESS_PERCENTAGE 
        print("open_ratio, task_success", open_ratio, task_success)

        with open(result_file_path, "a") as f:
            f.write(f"{OBJ_INDEX},{HANDLE_INDEX},{handle_path_str},{handle_joint_type},{handle_joint},{task_success},{open_ratio},{graps_pos},{grasp_rot},{v_desc}|{h_desc}\n")

        if SHOW_IMAGE:
            world.render()
            env.get_image().show()

        world.reset()
        controller.xforms.set_world_poses(positions=np.array([[0,0,0]]), orientations = np.array([[1, 0, 0, 0]])) # WXYZ
        for _ in range(30):
            world.step()

    # close object
    world.scene.remove_object(mobility_obj_name)
    world.render()

simulation_app.close()
