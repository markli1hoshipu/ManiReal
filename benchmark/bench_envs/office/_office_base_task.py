import os
import re
import sapien.core as sapien
from sapien.render import clear_cache as sapien_clear_cache
from sapien.utils.viewer import Viewer
import numpy as np
import gymnasium as gym
import pdb
import toppra as ta
import json
import transforms3d as t3d
from transforms3d.euler import euler2quat
from collections import OrderedDict
import torch, random

from bench_envs._bench_base_task import Bench_base_task
from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor

from copy import deepcopy
import subprocess
from pathlib import Path
import trimesh
import imageio
import glob


from envs._GLOBAL_CONFIGS import *

from typing import Optional, Literal

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


class Office_base_task(Bench_base_task):

    FURNITURE_NAMES = {"table", "wall", "shelf", "ground"}

    def __init__(self):
        pass

    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        """
        Initialization TODO
        - `self.FRAME_IDX`: The index of the file saved for the current scene.
        - `self.fcitx5-configtool`: Left gripper pose (close <=0, open >=0.4).
        - `self.ep_num`: Episode ID.
        - `self.task_name`: Task name.
        - `self.save_dir`: Save path.`
        - `self.left_original_pose`: Left arm original pose.
        - `self.right_original_pose`: Right arm original pose.
        - `self.left_arm_joint_id`: [6,14,18,22,26,30].
        - `self.right_arm_joint_id`: [7,15,19,23,27,31].
        - `self.render_fre`: Render frequency.
        """
        super().__init__() # need to fix this. right now this does nothing. it should be calling gym.Env.__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        np.random.seed(kwags.get("seed", 0))
        torch.manual_seed(kwags.get("seed", 0))
        # random.seed(kwags.get('seed', 0))
        self.seed = kwags.get("seed", 0)

        self.FRAME_IDX = 0
        self.task_name = kwags.get("task_name")
        self.save_dir = kwags.get("save_path", "data")
        self.ep_num = kwags.get("now_ep_num", 0)
        self.render_freq = kwags.get("render_freq", 10)
        self.data_type = kwags.get("data_type", None)
        self.save_data = kwags.get("save_data", False)
        self.dual_arm = kwags.get("dual_arm", True)
        self.eval_mode = kwags.get("eval_mode", False)
        self.sample_d = kwags.get("sample_d", "objects")
        self.enable_collision_metrics = kwags.get("enable_collision_metrics", False)

        self.need_topp = True  # TODO

        # Random
        random_setting = kwags.get("domain_randomization")
        self.random_background = random_setting.get("random_background", False)
        self.cluttered_table = random_setting.get("cluttered_table", False)
        self.clean_background_rate = random_setting.get("clean_background_rate", 1)
        self.random_head_camera_dis = random_setting.get("random_head_camera_dis", 0)
        self.random_table_height = random_setting.get("random_table_height", 0)
        self.random_light = random_setting.get("random_light", False)
        self.crazy_random_light_rate = random_setting.get("crazy_random_light_rate", 0)
        self.crazy_random_light = (0 if not self.random_light else np.random.rand() < self.crazy_random_light_rate)
        self.random_embodiment = random_setting.get("random_embodiment", False)  # TODO
        self.obstacle_height = random_setting.get("obstacle_height", "short")
        self.obstacle_density = random_setting.get("obstacle_density", 3)

        # Parse vision/object/language/background_plus perturbation flags
        # before setup_scene() so lighting ablation can consult them.
        self._parse_perturbations(random_setting)
        # Targets must keep using the baseline distribution (same asset ids
        # that check_success expects). Only obstacles swap to the OOD pool.
        self.obstacle_distribution = "object_ood" if self.unseen_obstacles else self.sample_d

        self.file_path = []
        self.plan_success = True
        self.step_lim = None
        self.fix_gripper = False
        self.setup_scene(**kwags)

        self.left_js = None
        self.right_js = None
        self.raw_head_pcl = None
        self.real_head_pcl = None
        self.real_head_pcl_color = None

        self.now_obs = {}
        self.take_action_cnt = 0
        self.eval_video_path = kwags.get("eval_video_save_dir", None)

        self.save_freq = kwags.get("save_freq")
        self.world_pcd = None

        self.key_objects = []
        self.size_dict = list()
        self.cluttered_objs = list()
        self.prohibited_area = {"table": [], "shelf0": [], "shelf1": []} # shelf 0 for lower shelf, shelf 1 for upper shelf
        # self.short_obstacles = [
        #     # "010_pen",
        #     # "015_laptop",
        #     "017_calculator",
        #     "021_cup",
        #     "022_cup-with-liquid",
        #     "023_tissue-box",
        #     "039_mug",
        #     "043_book",
        #     "047_mouse",
        #     "048_stapler",
        #     "055_small-speaker",
        #     "059_pencup",
        #     "063_tabletrashbin",
        #     "074_displaystand",
        #     "077_phone",
        #     "078_phonestand",
        #     "092_notebook",
        #     "099_fan",
        #     "100_seal",
        #     "116_keyboard",
        #     "120_plant"
        # ]
        # self.tall_obstacles = [
        #     "001_bottle",
        #     "089_globe",
        #     "095_glue",
        #     "097_screen",
        #     "098_speaker",
        #     "114_bottle",
        #     "119_mini-chalkboard"
        # ]
        self.record_cluttered_objects = list()  # record cluttered objects info
        self.cluttered_objects_info = get_cluttered_objects_info()

        self.eval_success = False
        # self.table_z_bias = (np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias)  # TODO
        self.table_z_bias = 0
        self.office_info = {
            "table_height": 0.74,
            "table_area":[1.2, 0.7], # x,y area 
            "table_lims": [],
            "shelf_heights":[0.9, 1.127], # heights of the shelf levels
            "shelf_area":[0.62, 0.26], # x,y area 
            "shelf_lims": [],
            "shelf_padding": 0.09, # required distance from edge of shelf for gripper to fit
            "file_holder_area":[0.22, 0.16], # x,y area 
            "file_holder_lims": [],
            "file_holder_heights": [0.82,0.942],
            "drawer_height": 0.76,
            "furn_x_v": { # x position of furniture for each arrangement version
                "shelf": [-0.24,0,0.24],
                "cabinet": [0.23,0.48,-0.48],
                "file_holder": [0.48,-0.48,-0.23],
            },
        }
        self.item_info = get_task_objects_config()
        self.target_objects_info = get_target_objects_subset("office", self.sample_d)
        self.need_plan = kwags.get("need_plan", True)
        self.left_joint_path = kwags.get("left_joint_path", [])
        self.right_joint_path = kwags.get("right_joint_path", [])
        self.left_cnt = 0
        self.right_cnt = 0

        self.instruction = None  # for Eval

        self.collision_list = [] # list of collision objects for curobo planner
        self.cuboid_collision_list = [] # list of cuboid collision objects for curobo planner
        self._init_collision_metrics()

        self.arr_v = np.random.choice([0,1,2]) # which version to use for furniture arrangement

        self.load_robot(**kwags)
        self.create_static_elements(table_xy_bias=table_xy_bias)
        self.load_camera(**kwags)
        self.robot.move_to_homestate()

        render_freq = self.render_freq
        self.render_freq = 0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.robot.set_origin_endpose()
        self.load_actors()
        self.add_gripper_operating_area()

        # Apply per-episode object perturbations on the freshly-loaded target.
        self._apply_target_texture()

        if self.cluttered_table:
            self.load_basic_office_items()
            self.get_cluttered_surfaces()

        # Lighting L3 (specular/shininess) must run after actors exist.
        self._apply_l3_specular()

        if self.enable_collision_metrics:
            self._build_collision_name_sets()  # build collision name sets for collision metrics

        is_stable, unstable_list = self.check_stable()
        if not is_stable:
            raise UnStableError(
                f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')
            # print(f'Objects is unstable in seed({kwags.get("seed", 0)}), unstable objects: {", ".join(unstable_list)}')

        self.update_world()

        if self.eval_mode:
            with open(os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", "_bench_eval_step_limit.yml"), "r") as f:
                try:
                    data = yaml.safe_load(f)
                    self.step_lim = data[self.task_name]
                except:
                    print(f"{self.task_name} not in step limit file, set to 1000")
                    self.step_lim = 1000

        # info
        self.info = dict()
        self.info["cluttered_table_info"] = self.record_cluttered_objects
        self.info["texture_info"] = {
            "wall_texture": self.wall_texture,
            "table_texture": self.table_texture,
            "floor_texture": self.floor_texture,
        }
        self.info["info"] = {}

        self.stage_success_tag = False

    def create_static_elements(self, table_xy_bias=[0, 0]):
        # create static furniture elements
        self.table_xy_bias = table_xy_bias
        wall_texture, table_texture, floor_texture = None, None, None
        table_height = self.office_info["table_height"] + self.table_z_bias

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = np.random.randint(0, file_count), np.random.randint(0, file_count)
            wall_texture = 43
            table_texture = 141
            floor_texture = 38

            self.wall_texture = f"seen/{wall_texture}"
            self.table_texture = f"seen/{table_texture}"
            self.floor_texture = f"seen/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture, self.table_texture, self.floor_texture = None, None, None
        
        # background_plus: resolve per-episode random RGB tints / material once
        bg_plus_on = getattr(self, "bg_plus_enabled", False)
        bg_wall_color = (1, 0.9, 0.9)
        bg_floor_color = (0.85, 0.85, 0.85)
        if bg_plus_on and getattr(self, "bg_plus_color_tint", True):
            lo, hi = self.bg_plus_tint_range
            bg_wall_color = tuple(float(np.random.uniform(lo, hi)) for _ in range(3))
            print(f"[Background+] wall tint={[round(c,2) for c in bg_wall_color]}")
        if bg_plus_on and getattr(self, "bg_plus_floor_texture", True):
            bg_floor_color = tuple(float(np.random.uniform(0.3, 1.0)) for _ in range(3))
            print(f"[Background+] floor tint={[round(c,2) for c in bg_floor_color]}")

        # floor--------------------------------------------------------
        self.floor_parts = []

        positions = [
            [1, 1, 0],    # top-right
            [-1, 1, 0],   # top-left
            [1, -1, 0],   # bottom-right
            [-1, -1, 0],  # bottom-left
        ]

        for i, pos in enumerate(positions):
            floor_part = create_visual_textured_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=bg_floor_color,
                name=f"floor_{i}",
                texture_id=self.floor_texture,
            )
            self.floor_parts.append(floor_part)

        # wall--------------------------------------------------------
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=bg_wall_color,
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=self.table_texture,
        )
        if bg_plus_on and getattr(self, "bg_plus_surface_material", True):
            try:
                m_lo, m_hi = self.bg_plus_metallic_range
                r_lo, r_hi = self.bg_plus_roughness_range
                metallic = float(np.random.uniform(m_lo, m_hi))
                roughness = float(np.random.uniform(r_lo, r_hi))
                for mat in (self.table.entity if hasattr(self.table, 'entity') else self.table).find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
                    try:
                        for part in mat.parts:
                            if part.material is not None:
                                part.material.metallic = metallic
                                part.material.roughness = roughness
                    except AttributeError:
                        if getattr(mat, 'material', None) is not None:
                            mat.material.metallic = metallic
                            mat.material.roughness = roughness
                print(f"[Background+] table metallic={metallic:.2f} roughness={roughness:.2f}")
            except Exception as e:
                print(f"[Background+] table material tweak failed: {e}")
        self.office_info["table_lims"] = [-self.office_info["table_area"][0]/2, -self.office_info["table_area"][1]/2, self.office_info["table_area"][0]/2, self.office_info["table_area"][1]/2]
        self.cuboid_collision_list.append({"name": "table", "dims": [1.2, 0.7, 0.002], "pose": [0,0,0.74,1,0,0,0]})
        # ------------------------------------------------------------
        depth = 0.28
        shelf_scale = [1.7,0.86,1.8] # length, height, depth
        pose = [self.office_info["furn_x_v"]["shelf"][self.arr_v],depth,table_height+0.27]
        self.shelf = create_glb_actor(
            scene=self.scene,
            pose=sapien.Pose(p=pose, q=[0.7071, 0.7071, 0.0, 0.0]),
            model_name="121_wall-shelf",
            scale=shelf_scale,
            convex=False,
            is_static=True,
            mass=2
        )
        self.collision_list.append({
            "actor": self.shelf,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/121_wall-shelf/cc0_wall_shelf_4.glb",
        })
        xmin = pose[0] - self.office_info["shelf_area"][0]/2
        xmax = pose[0] + self.office_info["shelf_area"][0]/2
        ymin = pose[1] - self.office_info["shelf_area"][1]/2
        ymax = pose[1] + self.office_info["shelf_area"][1]/2
        self.office_info["shelf_lims"] = [xmin, ymin, xmax, ymax]
        self.prohibited_area["table"].append([xmin-0.03, ymin-0.02, xmax+0.03, ymax+0.02])
        # ------------------------------------------------------------
        self.cabinet = create_sapien_urdf_obj(
            scene=self,
            modelname="036_cabinet",
            modelid=46653,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["cabinet"][self.arr_v],
                depth,
                table_height
                ], q=[0.7071, 0,0,0.7071]),
            fix_root_link=True,
        )
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.cabinet.set_mass(0.5)
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_0",
            "files": ["original-4.obj","original-7.obj"], # these are only the side panels of the cabinet. Drawer is added separately when needed
        })
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_3",
            "files": ["original-57.obj","original-62.obj"], # these are the top panel and handle. needed for collision checking
        })
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "link": "link_2",
            "files": ["original-34.obj", "original-41.obj"], # middle panel
        })
        # ------------------------------------------------------------
        self.wooden_box = create_actor(
            scene=self,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["file_holder"][self.arr_v],
                depth-0.06,
                0.813
                ], q=euler2quat(-np.pi/2,0,0, axes='sxyz')),
            modelname="042_wooden_box",
            convex=False,
            model_id=0,
            scale=[0.09,0.07,0.1],
            is_static=True,
        )
        # ------------------------------------------------------------
        self.file_holder = create_glb_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[
                self.office_info["furn_x_v"]["file_holder"][self.arr_v],
                depth-0.05,
                table_height+0.075
                ], q=[0.7071, 0.7071, 0, 0]),
            model_name="122_file-holder",
            scale=[0.38,0.7,0.4], # width, height, depth
            convex=False,
            is_static=True,
            mass=0.1
        )
        pose = self.file_holder.get_pose().p
        xmin = pose[0] - self.office_info["file_holder_area"][0]/2
        xmax = pose[0] + self.office_info["file_holder_area"][0]/2
        ymin = pose[1] - self.office_info["file_holder_area"][1]/2
        ymax = pose[1] + self.office_info["file_holder_area"][1]/2
        self.office_info["file_holder_lims"] = [xmin, ymin, xmax, ymax]
        self.prohibited_area["table"].append([xmin-0.01, ymin, xmax+0.01, ymax])
        self.collision_list.append({
            "actor": self.file_holder,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects_bench/122_file-holder/base.glb",
        })
        # ---------------- Texture randomization for furniture --------------------------------
        # change_object_texture(self, self.file_holder, str(np.random.randint(0, 3)),"file" ,refresh_render=True)
        # change_object_texture(self, self.shelf, str(np.random.randint(0, 3)),"shelf" ,refresh_render=True)
        # change_object_texture(self, self.cabinet, str(np.random.randint(0, 3)),"drawer" ,refresh_render=True)

    def load_basic_office_items(self):
        # load office items: items that are always placed as obstacles ie key obstacles
        entities = self.scene.get_all_actors()
        arts = self.scene.get_all_articulations()
        names = np.array([entity.get_name() for entity in entities])
        art_names = np.array([art.get_name() for art in arts])
        full_names = np.concatenate([names, art_names])

        size_dict = list()
        # if "015_laptop" not in full_names:
        #     # laptop_id = np.random.choice([9748,9912,9960,9968,9992,9996,10040,10098,10101,10125,10211])
        #     laptop_id = 9912
        #     # laptop_file = {
        #     #     9748: "original-97",
        #     #     9912: "new-4",
        #     #     9960: "new-0",
        #     #     9968: "new-8",
        #     #     9992: "new-1",
        #     #     9996: "new-4",
        #     #     10040: "new-5",
        #     #     10098: "new-4",
        #     #     10101: "new-1",
        #     #     10125: "new-7",
        #     #     10211: "new-1",
        #     # }
        #     success, self.laptop = rand_create_cluttered_actor(
        #         scene=self.scene,
        #         xlim=[-0.4, 0.4],
        #         ylim=[0.06],
        #         zlim=[self.office_info["table_height"]],
        #         modelname="015_laptop",
        #         modelid=laptop_id,
        #         modeltype="sapien_urdf",
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, np.pi / 10],
        #         qpos=[0.7, 0, 0, 0.7],
        #         size_dict=size_dict,
        #         obj_radius=0.06,
        #         z_offset=0,
        #         z_max=0.1,
        #         prohibited_area=self.prohibited_area["table"],
        #         constrained=False,
        #         is_static=False,
        #     )
        #     if not success:
        #         # print("Failed to load laptop")
        #         pass
        #     else:
        #         self.laptop.set_mass(0.1)
        #         limit = self.laptop.get_qlimits()[0]
        #         self.laptop.set_qpos([limit[0] + (limit[1] - limit[0]) * 0.9])
        #         self.add_prohibit_area(self.laptop, padding=0.01)
        #         self.collision_list.append({
        #             "actor": self.laptop,
        #             "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/015_laptop/9912/textured_objs/",
        #             "link": ["link_0", "link_1"],
        #             "files": ["original-5.obj"],
        #         })

        #     # # custom collision because laptop has too many meshes
        #     # cuboid_pose = self.laptop.get_pose().p.tolist() + [1, 0, 0, 0]
        #     # cuboid_pose[1] += 0.04
        #     # cuboid_pose[2] = self.office_info["table_height"] + 0.07
        #     # self.cuboid_collision_list.append({"name": "015_laptop", "dims": [0.2, 0.07, 0.14], "pose": cuboid_pose})
        # ------------------------------------------------------------
        if "plant" not in full_names:
            plant_id = 0
            success, self.plant = rand_create_cluttered_actor(
                scene=self.scene,
                xlim=[-0.5, 0.5],
                ylim=[0,0.11],
                zlim=[self.office_info["table_height"]],
                modelname="120_plant",
                modelid=plant_id,
                modeltype="glb",
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
                size_dict=size_dict,
                obj_radius=0.05,
                z_offset=self.cluttered_objects_info["120_plant"]["params"][f"{plant_id}"]["z_offset"],
                z_max=self.cluttered_objects_info["120_plant"]["params"][f"{plant_id}"]["z_max"],
                prohibited_area=self.prohibited_area["table"],
                constrained=False,
                is_static=False,
            )
            if not success:
                # print("Failed to load plant")
                pass
            else:
                self.plant.set_mass(1)
                pose = self.plant.get_pose().p
                self.prohibited_area["table"].append([pose[0]-0.03, pose[1]-0.03, pose[0]+0.03, pose[1]+0.03]) # manual because plant extents are incorrect
                self.collision_list.append({
                    "actor": self.plant,
                    "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/120_plant/collision/base{plant_id}.glb",
                })
    
    def get_cluttered_surfaces(self):
        # clutter surfaces with additional random obstacles
        # table ------------------------------------------------------
        xlim = [self.office_info["table_lims"][0], self.office_info["table_lims"][2]]
        ylim = [self.office_info["table_lims"][1], self.office_info["table_lims"][3]]
        zlim = [self.office_info["table_height"]]
        xlim[0] += self.table_xy_bias[0]
        xlim[1] += self.table_xy_bias[0]
        ylim[0] += self.table_xy_bias[1]
        ylim[1] += self.table_xy_bias[1]
        zlim = np.array(zlim) + self.table_z_bias
        
        # collect objects already on the scene
        task_objects_list = []
        for entity in self.scene.get_all_actors():
            actor_name = entity.get_name()
            if actor_name == "":
                continue
            if actor_name in ["table", "wall", "ground"]:
                continue
            task_objects_list.append(actor_name)

        cluttered_item_info, obj_names_short, obj_names_tall = get_obstacle_objects_subset(
            "office", self.obstacle_distribution, task_objects_list
        )

        self.clutter_surface_split(xlim, ylim, zlim, self.prohibited_area["table"], self.obstacle_density, cluttered_item_info, obj_names_short, obj_names_tall)
        # # shelves ----------------------------------------------------
        xlim = [self.office_info["shelf_lims"][0], self.office_info["shelf_lims"][2]]
        ylim = [self.office_info["shelf_lims"][1], self.office_info["shelf_lims"][3]]
        self.clutter_surface(xlim, ylim, [self.office_info["shelf_heights"][0]], self.prohibited_area["shelf0"], 5, cluttered_item_info, obj_names_short)
        self.clutter_surface(xlim, ylim, [self.office_info["shelf_heights"][1]], self.prohibited_area["shelf1"], 5, cluttered_item_info, obj_names_short)
    
    def add_extra_cameras(self):
        self.cameras.add_extra_cameras(f"{os.environ['BENCH_ROOT']}/bench_assets/embodiments/office_config.yml")
    
    def enable_drawer(self, enable: bool):
        files = ["original-23.obj", "original-24.obj", "original-18.obj", "original-34.obj", "original-41.obj", "original-57.obj", "original-62.obj"]
        names = [f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/{file}_{self.seed}" for file in files]
        self.enable_obstacle(enable, names)
    
    def disable_panel(self):
        # disable middle panel so that closing and opening dont throw a curobo error
        files = ["original-34.obj", "original-41.obj"]
        names = [f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/{file}_{self.seed}" for file in files]
        self.enable_obstacle(False, names)
    
    def enable_table(self, enable: bool):
        names = [f"table_[0, 0, 0.74, 1, 0, 0, 0]_{self.seed}"]
        self.enable_obstacle(enable, obb_names=names)

    def add_cabinet_collision(self):
        # adds cabinet drawer to curobo collision world. Adds it with the drawer state being open. 
        limit = self.cabinet.get_qlimits()[0]
        self.cabinet.set_qpos([limit[1],0,0]) # open drawer for extracting open pose
        self.collision_list.append({
            "actor": self.cabinet,
            "collision_path": f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/036_cabinet/46653/textured_objs/",
            "pose": self.cabinet.get_link_pose("link_1"),
            "files": ["original-23.obj", "original-24.obj", "original-18.obj"],
        })
        self.cabinet.set_qpos([limit[0],0,0]) # reset drawer to closed state for task rollout

        # prohibit area around opening space
        cabinet_pose = self.cabinet.get_pose().p
        cabinet_pose[1]-= 0.19  
        self.prohibited_area["table"].append([cabinet_pose[0]-0.11, cabinet_pose[1]-0.1, cabinet_pose[0]+0.11, cabinet_pose[1]+0.1])
        self.add_operating_area(cabinet_pose, width = 0.12, length = 0.4)
    
    def grasp_actor_from_table(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        # adds an intermediate step to disable table collision during the grasp for grasps that are close to the table
        _, actions = self.grasp_actor(actor = actor, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis, gripper_pos=gripper_pos, contact_point_id=contact_point_id)
        self.move((arm_tag, [actions[0]]))
        self.enable_table(enable=False)
        self.move((arm_tag, actions[1:]))