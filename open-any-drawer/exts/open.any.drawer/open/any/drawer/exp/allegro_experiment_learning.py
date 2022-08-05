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
result_file_path = "/home/yizhou/Research/Data/allegro_exp_learning.txt"
MODEL_PATH = "/home/yizhou/Research/temp0/fasterrcnn_resnet50_fpn.pth"
SHOW_IMAGE = True


import getpass
user = getpass.getuser()


from omni.isaac.kit import SimulationApp

# "/home/yizhou/Research/OpenAnyDrawer/scene0.usd" # 
usd_path = f"omniverse://localhost/Users/{user}/scene1.usd"
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
from hand_env import HandEnv
from task.checker import TaskChecker
from task.instructor import SceneInstructor
from omni.isaac.core.prims.xform_prim import XFormPrim

env = OpenEnv()
env.setup_viewport()

# env = HandEnv("/World/Hand/Bones/l_carpal_mid", "/World/Hand*/Bones/l_thumbSkeleton_grp/l_distalThumb_mid")
controller = HandEnv("/World/allegro*/allegro_mount", "/World/AnchorXform")

# init
world.reset()
controller.start()
world.scene.add(controller.robots)

if SHOW_IMAGE:
    world.render()
    env.get_image()

# load deep leanrning model
from exp.model import load_vision_model
model = load_vision_model(model_path = MODEL_PATH, model_name = "fasterrcnn_resnet50_fpn")

# iterate object index
for OBJ_INDEX in OBJ_INDEX_LIST[1:5]:
    OBJ_INDEX = int(OBJ_INDEX)

    # wrong index
    if OBJ_INDEX == 57 or OBJ_INDEX == 69:
        continue

    env.add_object(OBJ_INDEX, scale = 0.1)

    mobility_obj = XFormPrim("/World/Game/mobility")
    mobility_obj_name = mobility_obj.name

    world.scene.add(mobility_obj)
    world.reset()
    world.render()

    scene_instr = SceneInstructor()
    scene_instr.analysis()

    # export data and load model
    scene_instr.output_path = "/home/yizhou/Research/temp0/"
    scene_instr.export_data()
    omni.kit.commands.execute("DeletePrims", paths=["/Replicator"])
    scene_instr.model = model
    scene_instr.predict_bounding_boxes(image_path="/home/yizhou/Research/temp0/rgb_0.png")

    # if not valid
    if not scene_instr.is_obj_valid:
        print("object not valid: ", OBJ_INDEX)
        simulation_app.close()
        exit()
    
    # if no valid predicted boundbox
    if not scene_instr.is_pred_valid:
        with open(result_file_path, "a") as f:
            f.write(f"{OBJ_INDEX}, invalid prediction\n")

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
        for _ in range(500):
            world.step(render=True)         

        print("move to handle")
        # move to handle
        graps_pos[...,0] += 0.1
        controller.xforms.set_world_poses(graps_pos, grasp_rot)
        for _ in range(100):
            world.step(render=True)        

        print("close finger")
        # close finger
        finger_pos = np.array([
            [
            0, 0, 0, np.pi/2 + np.pi/18, 
            np.pi/5, np.pi/5, np.pi/5, 0,
            np.pi/5, np.pi/5, np.pi/5, np.pi/6,
            np.pi/5, np.pi/5, np.pi/5, np.pi/6,
            ],
        ])
        for i in range(120):
            controller.robots.set_joint_position_targets(finger_pos * i / 120) # 
            world.step(render=True)                   

        print("pull out")
        # pull out
        for i in range(300):
            graps_pos[...,0] -= 0.001
        #   env.robots.set_world_poses(graps_pos, grasp_rot)
            controller.xforms.set_world_poses(graps_pos, grasp_rot)
            controller.robots.set_joint_position_targets(finger_pos)
            world.step(render=True)

        # check task sucess
        open_ratio = task_checker.joint_checker.compute_percentage()
        if handle_joint_type == "PhysicsRevoluteJoint": # open a door the upper limit may reach 180 degree
            open_ratio *= 2
        task_success = open_ratio > SUCESS_PERCENTAGE 
        print("open_ratio, task_success", open_ratio, task_success)

        with open(result_file_path, "a") as f:
            f.write(f"{OBJ_INDEX},{HANDLE_INDEX},{handle_path_str},{handle_joint_type},{handle_joint},{task_success},{open_ratio},{graps_pos},{grasp_rot}\n")

        if SHOW_IMAGE:
            world.render()
            env.get_image().show()

        world.reset()
        controller.xforms.set_world_poses(positions=np.array([[0,0,0]]), orientations = np.array([[1, 0, 0, 0]])) # WXYZ

    # close object
    world.scene.remove_object(mobility_obj_name)
    world.render()

simulation_app.close()
