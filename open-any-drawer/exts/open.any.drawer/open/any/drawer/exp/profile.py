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
}