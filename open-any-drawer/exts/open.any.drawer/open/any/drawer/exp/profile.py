# params profile
import numpy as np


GRASP_PROFILES = {
    "allegro": {
        "usd_path": "omniverse://localhost/Users/yizhou/scene1.usd",
        "robot_path": "/World/allegro",
        "articulation_root": "/World/allegro*/allegro_mount",
        "offset":{
            "position_offset":{
                "vertical": [0.1,0,0.12],
                "horizontal": [0.1,-0.12,0],
            },
            "rotation":{
                "vertical": [0.38268, 0, 0, 0.92388], # XYZW
                "horizontal": [0.3036, 0.23296, -0.56242, 0.73296],
            },
        },
        "finger_pos": np.array([
            [
            0, 0, 0, np.pi/2 + np.pi/18, 
            np.pi/5, np.pi/5, np.pi/5, 0,
            np.pi/5, np.pi/5, np.pi/5, np.pi/6,
            np.pi/5, np.pi/5, np.pi/5, np.pi/6,
            ],
        ]), 
    },

    "shadowhand": {
        "usd_path": "omniverse://localhost/Users/yizhou/scene2.usd",
        "robot_path": "/World/shadow_hand",
        "articulation_root": "/World/shadow_hand*/robot0_hand_mount",
        "offset":{
            "position_offset":{
                "vertical": [0.04,0,0.42],
                "horizontal": [0.04,-0.42,0],
            },
            "rotation":{
                "vertical": [-0.5, 0.5, -0.5, 0.5], # XYZW
                "horizontal": [0, 0.70711, 0, 0.70711],
            },
        },
        "finger_pos": np.array([
            0.0
        ] * 24),
    },

    "frankahand": {
        "usd_path": "omniverse://localhost/Users/yizhou/scene4.usd",
        "robot_path": "/World/Franka",
        "articulation_root": "/World/Franka/panda_link8",
        "offset":{
            "position_offset":{
                "vertical": [0.04, -0.02, 0],
                "horizontal": [0.04,-0.01,-0.02],
            },
            "rotation":{
                "vertical": [-0.2706, -0.65328, 0.2706, 0.65328], # XYZW
                "horizontal": [0.2706, -0.65328, -0.2706, 0.65328],
            },
        },
        "finger_pos": np.array([[0.0, 0.0]]),
    },
}