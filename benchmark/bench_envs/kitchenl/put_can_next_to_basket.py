import os

import yaml

from bench_envs.kitchenl._kitchen_base_large import Kitchen_base_large
from bench_envs.utils.scene_gen_utils import get_actor_boundingbox, place_actor, point_to_box_distance, print_c
from envs.utils import *
import math
import numpy as np
import sapien
import transforms3d as t3d


class put_can_next_to_basket(Kitchen_base_large):
    MILK_BOX_MASS = 0.1
    MILK_BOX_SPAWN_Z_OFFSET = 0
    TABLE_WORLD_XY_JITTER = 0.05

    # Basket interior bounds in basket local frame.
    BASKET_X_BOUNDS = (-0.20, 0.20)
    BASKET_Y_BOUNDS = (-0.20, 0.20)
    BASKET_Z_BOUNDS = (-0.10, 0.25)

    # Success region multiplier around basket bounds.
    BASKET_EXPANSION_RATIO = 1

    # Left-arm motion tuning.
    APPROACH_DELTA_1 = dict(x=-0.22, y=0.22, z=-0.14)
    RETREAT_DELTA = dict(y=-0.07, z=0.02)
    GRASP_PRE_DIS = 0.07
    GRASP_DIS = 0.01
    GRASP_CLOSE_POS = 0.0

    def setup_demo(self, is_test: bool = False, **kwargs):
        self.can_modelname = "071_can"
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        self.can_model_ids = task_objs["objects"]["kitchenl"]["targets"][self.can_modelname]
        kwargs["scene_id"] = np.random.choice([0,1])
        kwargs["include_collision"] = True
        self.can_box_spawn_rot_deg = [0.45, 0.0, 90.0]
        kwargs["jitter_basket"] = False
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)


    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        if self.scene_id == 0:
            xlim = [-0.15, 0]
        else:
            xlim = [-0.45, -0.20]
        self.can, self.can_model_id, self.target_pose = \
        place_actor(self.can_modelname, self, col_thr=0.15, xlim=xlim, ylim=[-0.05], 
                    qpos=(90,0,0), object_bounds={}, task_objs=task_objs,
                     mass = 0.2, rotation=False, scene_name='kitchenl')

        self.add_prohibit_area(self.can, padding=0.04, area="table")
        bb_box = get_actor_boundingbox(self.basket_right.actor)      
        if self.scene_id == 0:
            y_place = np.random.uniform(low=bb_box[0][1]-0.1, high=bb_box[0][1]-0.05)
        else:
            y_place = np.random.uniform(low=bb_box[0][1]-0.15, high=bb_box[0][1]-0.1)

    
        self.des_obj_pose = [np.random.uniform(low=bb_box[0][0]+0.02, high=bb_box[1][0]),
             y_place, 0.75] + [1,0,0,0]
        
        self.add_prohibit_area(self.des_obj_pose, padding=0.0, area="table")

        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")
        

    def play_once(self):
        arm_tag = ArmTag("left")
        self.move(
            self.grasp_actor(
                self.can,
                arm_tag=arm_tag,
                pre_grasp_dis=self.GRASP_PRE_DIS,
                grasp_dis=self.GRASP_DIS
                # contact_point_id= self.contact_id
                # gripper_pos=self.GRASP_CLOSE_POS,
            )
        )
        self.attach_object(self.can, f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{self.can_modelname}/collision/base{self.can_model_id}.glb", str(arm_tag))

        self.move(
            self.place_actor(
                self.can,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain= "auto",
                pre_dis=0.01,
                dis=0.002,
            ))
        self.move(self.move_by_displacement(arm_tag, z = 0.04))
        self.info["info"] = {
            "{A}": f"{self.can_modelname}/base{self.can_model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        dist_thr = 0.15
        box_bb = get_actor_boundingbox(self.basket_right.actor)
        dist_to_box = point_to_box_distance(self.can.get_pose().p, box_bb[0], box_bb[1])

        return (dist_to_box < dist_thr
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

