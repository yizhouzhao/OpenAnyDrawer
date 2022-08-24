OBJ_INDEX_LIST = ['0', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25', '26', '27', '28', '29', '30', '31', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '52', '54', '55', '56', '58', '59', '60', '61', '62', '63', '64', '65', '66', '68', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '105', '106', '107', '108', '110', '111', '112', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '134', '135', '136', '137', '138', '139', '140', '142', '143', '144', '145', '146', '147', '148', '149', '151', '152', '153', '154', '156', '157', '158', '159', '160', '162', '163', '164', '165', '168', '169', '170', '171', '172', '173', '175', '176', '177', '179', '180', '182', '183', '184', '185', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197']

ALL_SEMANTIC_TYPES = [f"{v_desc}_{h_desc}_{cabinet_type}" for v_desc in ["", "bottom", "second-bottom", "middle", "second-top", "top"] for h_desc in ["", "right", "second-right", "middle", "second-left", "left"] for cabinet_type in ["drawer", "door"]]


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