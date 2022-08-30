import numpy as np
from PIL import Image

from exp.params import OBJ_INDEX_LIST, GRASP_PROFILES

ROBOT_NAME = "skeletonhand" # "shadowhand" # "allegro"
grasp_profile = GRASP_PROFILES[ROBOT_NAME]

SUCESS_PERCENTAGE = 20
print("SUCESS_PERCENTAGE: ", SUCESS_PERCENTAGE)
result_file_path = f"/home/yizhou/Research/Data/{ROBOT_NAME}_exp_learning823.txt"
MODEL_PATH = "/home/yizhou/Research/temp0/fasterrcnn_resnet50_fpn823.pth"

load_nucleus = True # nucleus loading
usd_path = "omniverse://localhost/Users/yizhou/scene3.usd" #grasp_profile["usd_path"]
SHOW_IMAGE = True


import getpass
user = getpass.getuser()


from omni.isaac.kit import SimulationApp    


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
from hand_common import HandBase
from render.utils import prim_random_color, LOOKS_PATH

from task.checker import TaskChecker
from task.instructor import SceneInstructor
from omni.isaac.core.prims.xform_prim import XFormPrim

env = OpenEnv(load_nucleus = load_nucleus)
env.add_camera()
env.setup_viewport()

# env = HandEnv("/World/Hand/Bones/l_carpal_mid", "/World/Hand*/Bones/l_thumbSkeleton_grp/l_distalThumb_mid")
# controller = HandEnv("/World/allegro*/allegro_mount", "/World/AnchorXform")

controller = HandBase(grasp_profile["articulation_root"], "/World/AnchorXform")
controller.grasp_profile = grasp_profile["offset"]

# init
world.reset()
controller.start()
world.scene.add(controller.robots)

# hide robot
hand_prim = world.scene.stage.GetPrimAtPath(grasp_profile["robot_path"])
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

    # randomize color

    # reset look in scene
    mat_look_prim = world.scene.stage.GetPrimAtPath(LOOKS_PATH)
    if mat_look_prim:
        omni.kit.commands.execute("DeletePrims", paths=[LOOKS_PATH])

    world.step(render = False)

    scene_instr = SceneInstructor()
    scene_instr.analysis()

    handle_num = len(list(scene_instr.valid_handle_list.keys()))

    for HANDLE_INDEX in range(handle_num):
        handle_path_str = list(scene_instr.valid_handle_list.keys())[HANDLE_INDEX]
        prim_random_color(handle_path_str)
        
    world.scene.add(mobility_obj)
    world.reset()

    world.render()
    world.render()
    
    image_array =env.get_image(return_array=True)

    if SHOW_IMAGE:
        world.render()
        env.get_image().show()

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
        # print("handle_path_str, handle_joint_type, handle_joint, rel_direction", handle_path_str, handle_joint_type, handle_joint, handle_rel_direciton)
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
            world.step(render=SHOW_IMAGE)         

        print("move to handle")
        # move to handle
        graps_pos[...,0] += 0.1
        controller.xforms.set_world_poses(graps_pos, grasp_rot)
        for _ in range(100):
            world.step(render=SHOW_IMAGE)     

                
        # close finger
        print("close finger")
        finger_pos = grasp_profile["finger_pos"]

        if ROBOT_NAME == "allegro":   
            for i in range(120):
                controller.robots.set_joint_position_targets(finger_pos * i / 120) # 
                world.step(render=SHOW_IMAGE)       

        elif ROBOT_NAME == "frankahand":      
            for i in range(100):
                finger_pos -= 0.01
                controller.robots.set_joint_position_targets(finger_pos) # 
                world.step(render=SHOW_IMAGE) 

        elif ROBOT_NAME == "shadowhand": 
            dof_pos = finger_pos
            for i in range(60):
                # thumb
                dof_pos[6] += 0.01
                dof_pos[11] += 0.02
                # dof_pos[16] += 0.01
                dof_pos[21] += -0.01
                
                
                dof_pos[7] += 0.01
                dof_pos[8] += 0.01
                dof_pos[9] += 0.01
                # dof_pos[14] += 0.01
                
                dof_pos[12] += 0.01
                dof_pos[13] += 0.01
                dof_pos[14] += 0.01
                
                dof_pos[17] += 0.01
                dof_pos[18] += 0.01
                dof_pos[19] += 0.01
                
                # pinky
                dof_pos[15] += 0.01
                dof_pos[20] += 0.01
                dof_pos[22] += 0.01
                
                controller.robots.set_joint_position_targets(dof_pos) # 
                world.step(render=SHOW_IMAGE)   

        elif ROBOT_NAME == "skeletonhand": 
            # close finger
            for i in range(120):
                i  = i / 4
                dof_pos = np.array([
                    [ i * 0.03,  i * 0.04, 
                    i * 0.01,  -i * 0.04,  
                    i * 0.005, -i * 0.04, 
                    -i * 0.02, -i * 0.04,  
                    -i * 0.01, -i * 0.04,  
                    -i * 0.02,  -i * 0.03,  -i * 0.03,  -i * 0.03,  -i * 0.03,
                    -i * 0.02,  -i * 0.03,  -i * 0.03,  -i * 0.03,  -i * 0.03, 
                    ],
                ])

                # pos = np.random.randn(2,25)
                controller.robots.set_joint_position_targets(dof_pos) # 
                world.step(render=SHOW_IMAGE)  

        print("pull out")
        # pull out
        if ROBOT_NAME == "allegro": 
            for i in range(300):
                graps_pos[...,0] -= 0.001
            #   env.robots.set_world_poses(graps_pos, grasp_rot)
                controller.xforms.set_world_poses(graps_pos, grasp_rot)
                controller.robots.set_joint_position_targets(finger_pos)
                world.step(render=SHOW_IMAGE)

        elif ROBOT_NAME == "frankahand": 
            for i in range(300):
                graps_pos[...,0] -= 0.001
                controller.xforms.set_world_poses(graps_pos, grasp_rot)
                controller.robots.set_joint_position_targets(finger_pos)
                finger_pos += 0.015
                world.step(render=SHOW_IMAGE)

        elif ROBOT_NAME == "shadowhand": 
            # pull out
            for i in range(300):
                graps_pos[...,0] -= 0.001
            #   env.robots.set_world_poses(graps_pos, grasp_rot)
                controller.xforms.set_world_poses(graps_pos, grasp_rot)
                controller.robots.set_joint_position_targets(dof_pos)
                dof_pos *= 0.997
                # print(dof_pos)

                world.step(render=SHOW_IMAGE)
        
        elif ROBOT_NAME == "skeletonhand": 
            # pull out
            for i in range(200):
                graps_pos[...,0] -= 0.001
            #   env.robots.set_world_poses(graps_pos, grasp_rot)
                controller.xforms.set_world_poses(graps_pos, grasp_rot)
                controller.robots.set_joint_position_targets(dof_pos)

                world.step(render=SHOW_IMAGE)

            dof_pos /= 1.5
            # pull out furthur
            for i in range(100):
                graps_pos[...,0] -= 0.001
            #   env.robots.set_world_poses(graps_pos, grasp_rot)
                controller.xforms.set_world_poses(graps_pos, grasp_rot)
                controller.robots.set_joint_position_targets(dof_pos)
                world.step(render=SHOW_IMAGE)
        

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
