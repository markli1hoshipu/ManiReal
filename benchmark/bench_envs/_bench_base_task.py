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
from collections import OrderedDict
import torch, random

from envs.utils import *
from bench_envs.utils import *
import math
from envs.robot import Robot
from envs.camera import Camera
from envs.utils.actor_utils import Actor, ArticulationActor
from envs._base_task import Base_Task

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


class Bench_base_task(Base_Task):
    """
    Base task for all benchmark tasks. Mimics robotwin base task, with some functionality changes
    """
    FURNITURE_NAMES = {"table", "wall", "ground"}
    # Gripper links excluded from robot-to-furniture/static collision metrics (expected contact during manipulation)
    GRIPPER_LINK_NAMES = {"fr_link7", "fr_link8", "fl_link7", "fl_link8"}
    # Collision force threshold (N): ignore contacts with avg force below this.
    COLLISION_FORCE_THRESHOLD_N = 10.0
    # Static object pose thresholds: only count robot/target-to-static collisions when
    # the static object has moved beyond these from the previous step.
    STATIC_OBJECT_POSITION_THRESHOLD_M = 0.01   # 1 cm
    STATIC_OBJECT_ORIENTATION_THRESHOLD_RAD = 0.1  # ~5.7 deg

    def __init__(self):
        pass

    # =========================================================== Init Task Env ===========================================================
    def _init_task_env_(self, table_xy_bias=[0, 0], table_height_bias=0, **kwags):
        pass


    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("optix")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.timestep = kwargs.get("timestep", 1 / 250)
        self.scene.set_timestep(self.timestep)
        # Impulse threshold = force_threshold * timestep (impulse in N·s)
        self.collision_impulse_threshold = max(
            self.COLLISION_FORCE_THRESHOLD_N * self.timestep,
            1e-6,  # floor to avoid numerical noise
        )
        # add ground to scene
        self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 2),
            kwargs.get("dynamic_friction", 1),
            kwargs.get("restitution", 0),
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
        point_lights = kwargs.get("point_lights", [[[1, 0, 1.8], [1, 1, 1]], [[-1, 0, 1.8], [1, 1, 1]]])

        apply_lighting_ablation = getattr(self, "apply_lighting_ablation", False)
        if apply_lighting_ablation:
            # Per-episode L1/L2/L3/L4 toggles mirror robotwin-plus. L3 (specular)
            # is applied later once actors exist; here we only set lights.
            apply_l1 = getattr(self, "_l1_enabled", True)
            apply_l4 = getattr(self, "_l4_enabled", True) and (np.random.rand() < 0.5)
            apply_l2 = getattr(self, "_l2_enabled", True) and apply_l4
            l1_range = getattr(self, "_l1_range", [0.4, 1.8])

            self.direction_light_lst = []
            for direction_light in direction_lights:
                direction, color = list(direction_light[0]), list(direction_light[1])
                if apply_l1:
                    color = [float(np.random.uniform(l1_range[0], l1_range[1])) for _ in range(3)]
                if apply_l2:
                    theta = np.random.uniform(np.deg2rad(8), np.deg2rad(82))
                    phi = np.random.uniform(0, 2 * np.pi)
                    direction = [
                        float(np.sin(theta) * np.cos(phi)),
                        float(np.sin(theta) * np.sin(phi)),
                        float(np.cos(theta)),
                    ]
                self.direction_light_lst.append(
                    self.scene.add_directional_light(direction, color, shadow=apply_l4))

            self.point_light_lst = []
            for point_light in point_lights:
                pos, color = list(point_light[0]), list(point_light[1])
                if apply_l1:
                    color = [float(np.random.uniform(l1_range[0], l1_range[1])) for _ in range(3)]
                self.point_light_lst.append(self.scene.add_point_light(pos, color, shadow=apply_l4))
            print(f"[Lighting] L1={apply_l1} L2={apply_l2} L3=(deferred) L4={apply_l4}")
        else:
            self.direction_light_lst = []
            for direction_light in direction_lights:
                if self.random_light:
                    direction_light[1] = [
                        np.random.rand(),
                        np.random.rand(),
                        np.random.rand(),
                    ]
                self.direction_light_lst.append(
                    self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow))
            self.point_light_lst = []
            for point_light in point_lights:
                if self.random_light:
                    point_light[1] = [np.random.rand(), np.random.rand(), np.random.rand()]
                self.point_light_lst.append(self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow))

        # initialize viewer with camera position and orientation
        if self.render_freq:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(
                x=kwargs.get("camera_xyz_x", 0.4),
                y=kwargs.get("camera_xyz_y", 0.22),
                z=kwargs.get("camera_xyz_z", 1.5),
            )
            self.viewer.set_camera_rpy(
                r=kwargs.get("camera_rpy_r", 0),
                p=kwargs.get("camera_rpy_p", -0.8),
                y=kwargs.get("camera_rpy_y", 2.45),
            )

    def create_static_elements(self, table_xy_bias=[0, 0], table_height=0.74):
        pass

    # ==================================================================
    # Perturbation parsing + application helpers.
    # Called by subclasses (Office/Study/Kitchen base tasks) from their
    # own _init_task_env_ so vision / object / language / background_plus
    # flags get wired uniformly.
    # ==================================================================
    def _parse_perturbations(self, random_setting):
        random_setting = random_setting or {}
        noise_types = ['motion', 'gaussian', 'zoom', 'fog', 'glass']

        # Vision — lighting
        vision = random_setting.get("vision_perturbation", {}) or {}
        lighting = vision.get("lighting", {}) or {}
        self.apply_lighting_ablation = bool(lighting.get("enabled", False))
        self._l1_enabled = bool(lighting.get("l1", True))
        self._l2_enabled = bool(lighting.get("l2", True))
        self._l3_enabled = bool(lighting.get("l3", True))
        self._l4_enabled = bool(lighting.get("l4", True))
        self._l1_range = lighting.get("l1_range", [0.4, 1.8])

        # Vision — blur (per-episode noise type; per-frame CV ops applied
        # in envs/_base_task.py:get_obs).
        blur = vision.get("blur", {}) or {}
        if bool(blur.get("enabled", False)):
            self.blur_perturb_enabled = True
            forced = blur.get("force_type")
            if forced and forced in noise_types:
                self.current_noise_type = forced
            else:
                self.current_noise_type = noise_types[self.ep_num % len(noise_types)]
            # Severity 2/5 gives s=0.25 which is visually meaningful but not
            # destructive; overall scaled by YAML `strength`.
            self.current_severity = 2
            self.current_s = (self.current_severity - 1) / 4.0
            self.blur_strength = float(blur.get("strength", 1.0))
            if self.current_noise_type == 'zoom':
                zoom_min = 1.00
                zoom_max = 1.1 + self.current_s * (1.56 - 1.11)
                self.zoom_factor = float(np.random.uniform(zoom_min, zoom_max))
            else:
                self.zoom_factor = None
            print(f"[Vision] blur={self.current_noise_type} s={self.current_s:.2f} strength={self.blur_strength}")
        else:
            self.blur_perturb_enabled = False
            self.current_noise_type = None

        # Vision — pixel shift (per-frame; randomized inside consumer loop).
        shift = vision.get("pixel_shift", {}) or {}
        self.pixel_shift_enabled = bool(shift.get("enabled", False))
        self.pixel_shift_max = float(shift.get("max_shift", 5))
        self.pixel_shift_strength = float(shift.get("strength", 1.0))

        # Object perturbation
        obj = random_setting.get("object_perturbation", {}) or {}
        tgt_tex = obj.get("target_texture", {}) or {}
        self.target_texture_enabled = bool(tgt_tex.get("enabled", False))
        self.target_texture_source = str(tgt_tex.get("texture_source", "seen"))
        self.unseen_obstacles = bool(obj.get("unseen_obstacles", False))

        # Background_plus (B1+/B2) — ported from robotwin-plus.
        bg_plus = random_setting.get("background_plus", {}) or {}
        self.bg_plus_enabled = bool(bg_plus.get("enabled", False))
        self.bg_plus_color_tint = bool(bg_plus.get("color_tint", True))
        self.bg_plus_tint_range = bg_plus.get("tint_range", [0.5, 1.5])
        self.bg_plus_surface_material = bool(bg_plus.get("surface_material", True))
        self.bg_plus_metallic_range = bg_plus.get("metallic_range", [0.0, 0.8])
        self.bg_plus_roughness_range = bg_plus.get("roughness_range", [0.05, 0.95])
        self.bg_plus_floor_texture = bool(bg_plus.get("floor_texture", True))

        # Language
        lang = random_setting.get("language_perturbation", {}) or {}
        self.language_perturbation_enabled = bool(lang.get("enabled", False))
        self._instruction_bank_path = lang.get("instruction_bank")

    def _apply_l3_specular(self):
        """Apply L3 specular/shininess variation to all actors in the scene.
        Must be called after load_actors. Robotwin-plus reference:
        See _base_task.py:478-492.
        """
        if not getattr(self, "apply_lighting_ablation", False):
            return
        if not getattr(self, "_l3_enabled", True):
            return
        if np.random.rand() >= 0.5:
            return
        specular_strength = float(np.random.uniform(0.3, 6.0))
        shininess = float(np.random.uniform(10, 250))
        for actor in self.scene.get_all_actors():
            mats = actor.get_materials() if hasattr(actor, 'get_materials') else []
            for mat in mats:
                if mat is None:
                    continue
                try:
                    mat.set_specular(specular_strength)
                    mat.set_shininess(shininess)
                except Exception:
                    pass
        print(f"[Lighting] L3 spec={specular_strength:.2f} shine={shininess:.0f}")

    def _apply_target_texture(self):
        """Per-episode swap of the target object's base-color texture with a
        random background-texture PNG. Gated on object_perturbation.target_texture.
        """
        if not getattr(self, "target_texture_enabled", False):
            return
        target = getattr(self, "target_obj", None)
        if target is None:
            return
        src = getattr(self, "target_texture_source", "seen")
        if src == "background":
            src = "seen"
        tex_dir = f"./assets/background_texture/{src}"
        if not os.path.isdir(tex_dir):
            return
        pngs = [f for f in os.listdir(tex_dir) if f.endswith(".png")]
        if not pngs:
            return
        tex_path = os.path.join(tex_dir, np.random.choice(pngs))
        try:
            # In this codebase target_obj is the Actor wrapper; the underlying
            # sapien.Entity lives on `.actor`. Fall back to `.entity` or the
            # object itself for other shapes.
            entity = (getattr(target, "actor", None)
                      or getattr(target, "entity", None)
                      or target)
            rbc = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if rbc is None:
                return
            tex2d = sapien.render.RenderTexture2D(tex_path)
            for shape in rbc.render_shapes:
                try:
                    for part in shape.parts:
                        mat = part.material
                        if mat is None:
                            continue
                        try:
                            mat.set_base_color_texture(tex2d)
                            mat.set_base_color([1, 1, 1, 1])
                        except Exception:
                            pass
                except AttributeError:
                    try:
                        mat = shape.material
                        if mat is not None:
                            mat.set_base_color_texture(tex2d)
                            mat.set_base_color([1, 1, 1, 1])
                    except Exception:
                        pass
            print(f"[Object] target_texture -> {os.path.basename(tex_path)}")
        except Exception as e:
            print(f"[Object] target_texture failed: {e}")

    def _maybe_apply_language_perturbation(self):
        """If enabled, pick a random instruction from instruction_bank.json for
        the current task_name and set it as the active instruction. Records the
        result to self.info so collect_data.py persists it into scene_info.json.
        """
        if not getattr(self, "language_perturbation_enabled", False):
            return None
        bank_path = getattr(self, "_instruction_bank_path", None)
        if not bank_path:
            return None
        if not os.path.isabs(bank_path):
            # Try BENCH_ROOT (where bench_task_config lives) then ROBOTWIN_ROOT.
            for root in (os.environ.get("BENCH_ROOT"), os.environ.get("ROBOTWIN_ROOT"), "."):
                if not root:
                    continue
                # Config path may already start with 'benchmark/…' when BENCH_ROOT
                # is the 'benchmark' dir, so try both joined and stripped forms.
                candidates = [os.path.join(root, bank_path)]
                if bank_path.startswith("benchmark/"):
                    candidates.append(os.path.join(root, bank_path[len("benchmark/"):]))
                for c in candidates:
                    if os.path.exists(c):
                        bank_path = c
                        break
                else:
                    continue
                break
        if not os.path.exists(bank_path):
            print(f"[Language] bank not found at {bank_path}")
            return None
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                bank = json.load(f)
        except Exception as e:
            print(f"[Language] failed to load bank: {e}")
            return None
        pool = bank.get(self.task_name, [])
        if not pool:
            return None
        instruction = str(np.random.choice(pool))
        self.set_instruction(instruction=instruction)
        if not isinstance(getattr(self, "info", None), dict):
            self.info = {}
        self.info["language_perturbation"] = {
            "instruction": instruction,
            "bank": bank_path,
        }
        print(f"[Language] {self.task_name} -> '{instruction[:60]}'")
        return instruction

    def get_cluttered_surfaces(self):
        pass
    
    def clutter_surface_split(self, xlim, ylim, zlim, prohibited_area, obstacle_count, cluttered_item_info, obj_names_short, obj_names_tall):
        """
        Produce clutter on a given surface from 2 object name pools
        """
        # # for viewing area estimation
        # for area in prohibited_area:
        #     x_min = area[0]
        #     x_max = area[2]
        #     y_min = area[1]
        #     y_max = area[3]
        #     half_size = [(x_max-x_min)/2, (y_max-y_min)/2, 0.0005]
        #     target = create_box(
        #         scene=self,
        #         pose=sapien.Pose([x_min+half_size[0], y_min+half_size[1], 0.74], [1,0,0,0]),
        #         half_size=half_size,
        #         color=(1, 0, 0),
        #         name=f"_collision",
        #         is_static=True,
        #     )
        # record cluttered objects
        self.record_cluttered_objects = []
        self.size_dict = []

        if np.random.rand() < self.clean_background_rate:
            return

        success_count = 0
        max_try = 50
        trys = 0

        # Track which specific model ids have been placed per object name
        placed_objects = {name: [] for name in cluttered_item_info.keys()}

        # Precompute desired counts by group (may not be reached if placement fails)
        short_target = int(round(0.3 * obstacle_count))
        tall_target = obstacle_count - short_target

        short_count = 0
        tall_count = 0

        # Build flat lists for sampling indices within each group
        obj_names_short = list(obj_names_short)
        obj_names_tall = list(obj_names_tall)

        # If one group is empty, fall back to the other
        if not obj_names_short and not obj_names_tall:
            return

        while success_count < obstacle_count and trys < max_try:
            # Decide which group to sample from for this attempt
            if not obj_names_short:
                group = "tall"
            elif not obj_names_tall:
                group = "short"
            else:
                # Prefer to fill up to targets with 30% short / 70% tall
                if short_count < short_target and tall_count < tall_target:
                    # sample with 0.3 / 0.7 probability
                    group = "short" if np.random.rand() < 0.3 else "tall"
                elif short_count < short_target:
                    group = "short"
                elif tall_count < tall_target:
                    group = "tall"
                else:
                    # both groups reached their nominal target; continue with 0.3/0.7 split
                    group = "short" if np.random.rand() < 0.3 else "tall"

            if group == "short":
                obj_list = obj_names_short
            else:
                obj_list = obj_names_tall

            if not obj_list:
                break

            obj = np.random.randint(len(obj_list))
            obj_name = obj_list[obj]

            # Randomly choose an index within available ids for this object
            ids_for_obj = cluttered_item_info[obj_name]["ids"]

            rand_idx = np.random.randint(len(ids_for_obj))
            obj_idx = ids_for_obj[rand_idx]

            if obj_idx in placed_objects.get(obj_name, []):
                trys += 1
                continue

            obj_radius = cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
            obj_offset = cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
            obj_maxz = cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]
            scale = cluttered_item_info[obj_name]["params"][obj_idx]["scale"]

            success, self.cluttered_obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                modelname=obj_name,
                modelid=obj_idx,
                scale=scale,
                modeltype=cluttered_item_info[obj_name]["type"],
                rotate_rand=True,
                rotate_lim=[0, 0, math.pi],
                size_dict=self.size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=prohibited_area,
                is_static=False,
                constrained=False,
            )
            if not success or self.cluttered_obj is None:
                trys += 1
                continue

            self.cluttered_obj.set_name(f"{obj_name}")

            # manage stability as distractors
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1

            if group == "short":
                short_count += 1
            else:
                tall_count += 1

            self.record_cluttered_objects.append(
                {"object_type": obj_name, "object_index": obj_idx}
            )
            placed_objects[obj_name].append(obj_idx)

            # add to collision list
            if cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append({
                "actor": self.cluttered_obj,
                "collision_path": path,
            })

            # # for viewing radius estimation
            # half_size = [obj_radius, obj_radius, 0.0005]
            # pose = self.cluttered_obj.get_pose()
            # pose.q = [1,0,0,0]
            # target = create_box(
            #     scene=self,
            #     pose=pose,
            #     half_size=half_size,
            #     color=(1, 0, 0),
            #     name=f"{obj_name}_collision",
            #     is_static=True,
            # )
        
        if success_count < obstacle_count:
            print(f"Warning: Only {success_count} cluttered objects are placed on the surface.")

        self.size_dict = None
        self.cluttered_objs = []

    def clutter_surface(self, xlim, ylim, zlim, prohibited_area, obstacle_count, cluttered_item_info, obj_names):
        """
        Produce clutter on a given surface
        """
        # # for viewing area estimation
        # for area in prohibited_area:
        #     x_min = area[0]
        #     x_max = area[2]
        #     y_min = area[1]
        #     y_max = area[3]
        #     half_size = [(x_max-x_min)/2, (y_max-y_min)/2, 0.0005]
        #     target = create_box(
        #         scene=self,
        #         pose=sapien.Pose([x_min+half_size[0], y_min+half_size[1], zlim[0]], [1,0,0,0]),
        #         half_size=half_size,
        #         color=(1, 0, 0),
        #         name=f"_collision",
        #         is_static=True,
        #     )

        # record cluttered objects
        self.record_cluttered_objects = []
        self.size_dict = []

        if np.random.rand() < self.clean_background_rate:
            return

        success_count = 0
        max_try = 50
        trys = 0

        # Track which specific model ids have been placed per object name
        placed_objects = {name: [] for name in cluttered_item_info.keys()}

        obj_names = list(obj_names)
        if not obj_names:
            return

        while success_count < obstacle_count and trys < max_try:
            obj = np.random.randint(len(obj_names))
            obj_name = obj_names[obj]

            # Randomly choose an index within available ids for this object
            ids_for_obj = cluttered_item_info[obj_name]["ids"]
            rand_idx = np.random.randint(len(ids_for_obj))
            obj_idx = ids_for_obj[rand_idx]

            if obj_idx in placed_objects.get(obj_name, []):
                trys += 1
                continue

            obj_radius = cluttered_item_info[obj_name]["params"][obj_idx]["radius"]
            obj_offset = cluttered_item_info[obj_name]["params"][obj_idx]["z_offset"]
            obj_maxz = cluttered_item_info[obj_name]["params"][obj_idx]["z_max"]
            scale = cluttered_item_info[obj_name]["params"][obj_idx]["scale"]

            success, self.cluttered_obj = rand_create_cluttered_actor(
                self.scene,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                modelname=obj_name,
                modelid=obj_idx,
                scale=scale,
                modeltype=cluttered_item_info[obj_name]["type"],
                rotate_rand=True,
                rotate_lim=[0, 0, math.pi],
                size_dict=self.size_dict,
                obj_radius=obj_radius,
                z_offset=obj_offset,
                z_max=obj_maxz,
                prohibited_area=prohibited_area,
                is_static=False,
                constrained=False,
            )
            if not success or self.cluttered_obj is None:
                trys += 1
                continue

            self.cluttered_obj.set_name(f"{obj_name}")

            # manage stability as distractors
            self.stabilize_object(self.cluttered_obj)

            self.cluttered_objs.append(self.cluttered_obj)
            pose = self.cluttered_obj.get_pose().p.tolist()
            pose.append(obj_radius)
            self.size_dict.append(pose)
            success_count += 1

            self.record_cluttered_objects.append(
                {"object_type": obj_name, "object_index": obj_idx}
            )
            placed_objects[obj_name].append(obj_idx)

            # add to collision list
            if cluttered_item_info[obj_name]["type"] == "urdf":
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/objaverse/{obj_name}/{obj_idx}/coacd_collision.obj"
            else:
                path = f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/{obj_name}/collision/base{obj_idx}.glb"
            self.collision_list.append({
                "actor": self.cluttered_obj,
                "collision_path": path,
            })

        if success_count < obstacle_count:
            print(f"Warning: Only {success_count} cluttered objects are placed on the surface.")

        self.size_dict = None
        self.cluttered_objs = []
    
    def stabilize_object(self, object):
        # object.set_mass(1)
        rb = object.actor.components[1]
        try:
            rb.set_linear_damping(5.0)
            rb.set_angular_damping(20.0)
        except:
            pass

    # =========================================================== Collision Metrics ===========================================================

    def _init_collision_metrics(self):
        """Reset collision tracking state. Call early in _init_task_env_ before load_actors()."""
        self.target_object_names: set[str] = set()
        self.collision_metrics = {
            "robot_to_furniture": 0,
            "robot_to_static_object": 0,
            "target_to_static_object": 0,
            "robot_to_furniture_steps": 0,
            "robot_to_static_object_steps": 0,
            "target_to_static_object_steps": 0,
        }
        self.filtered_contacts_for_log = []

    def _get_target_object_names(self) -> set[str]:
        """Return the names of target objects for this task.
        Must be overridden by every concrete task subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_target_object_names()"
        )

    def _build_collision_name_sets(self):
        """
        Build name sets for collision detection. Must be called after all actors
        (robot, furniture, target objects, static objects, clutter) are loaded.
        """
        self.robot_link_names = set(
            link.get_name() for link in
            self.robot.left_entity.get_links() + self.robot.right_entity.get_links()
        )
        self.furniture_names = set(self.FURNITURE_NAMES)
        self.target_object_names = self._get_target_object_names()

        all_actor_names = {
            entity.get_name() for entity in self.scene.get_all_actors()
            if entity.get_name()
        }
        
        self.static_object_names = all_actor_names - (self.furniture_names | self.target_object_names)

        # Store previous-step poses for static objects (used to filter collisions by step-to-step pose change)
        self.static_object_pose_prev = {}
        for entity in self.scene.get_all_actors():
            name = entity.get_name()
            if name in self.static_object_names:
                pose = entity.get_pose()
                self.static_object_pose_prev[name] = (
                    np.array(pose.p, dtype=np.float64),
                    np.array(pose.q, dtype=np.float64),
                )

    def _static_object_has_significant_pose_change(self, name: str) -> bool:
        """Return True if the static object has moved/rotated beyond thresholds since the previous step."""
        prev = self.static_object_pose_prev.get(name)
        if prev is None:
            return True  # unknown object, count the collision
        prev_p, prev_q = prev
        entity = next((e for e in self.scene.get_all_actors() if e.get_name() == name), None)
        if entity is None:
            return True
        curr = entity.get_pose()
        curr_p = np.array(curr.p, dtype=np.float64)
        curr_q = np.array(curr.q, dtype=np.float64)
        pos_delta = float(np.linalg.norm(curr_p - prev_p))
        qdot = abs(float(np.dot(curr_q, prev_q)))
        ang_delta = 2 * np.arccos(min(1.0, qdot))
        return (
            pos_delta >= self.STATIC_OBJECT_POSITION_THRESHOLD_M
            or ang_delta >= self.STATIC_OBJECT_ORIENTATION_THRESHOLD_RAD
        )

    def check_collisions(self):
        """
        Query PhysX contacts after scene.step() and accumulate collision counts.
        Categories:
          - robot_to_furniture:      robot link <-> furniture (table, wall, shelf, ground)
          - robot_to_static_object:  robot link <-> movable static objects (screen, clutter, etc.)
          - target_to_static_object: target object <-> movable static objects (held obj bumping things)
        Furniture: only count contacts with impulse above threshold.
        Static objects: only count when static object has significant pose change from previous step (e.g. knocked over, fallen).
        Populates self.filtered_contacts_for_log with contact points that passed filters (for debug/logging).
        """
        contacts = self.scene.get_contacts()
        self.filtered_contacts_for_log = []

        step_has_furniture = False
        step_has_static = False
        step_has_target_static = False

        for contact in contacts:
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name

            has_impulse = any(
                np.linalg.norm(point.impulse) > self.collision_impulse_threshold
                for point in contact.points
            )

            is_robot_0 = name0 in self.robot_link_names
            is_robot_1 = name1 in self.robot_link_names
            is_gripper_0 = name0 in self.GRIPPER_LINK_NAMES
            is_gripper_1 = name1 in self.GRIPPER_LINK_NAMES
            is_furniture_0 = name0 in self.furniture_names
            is_furniture_1 = name1 in self.furniture_names
            is_target_0 = name0 in self.target_object_names
            is_target_1 = name1 in self.target_object_names
            is_static_0 = name0 in self.static_object_names
            is_static_1 = name1 in self.static_object_names

            count_furniture = False
            count_static = False
            count_target_static = False

            # Furniture: require impulse (actual force exchange); exclude gripper links (expected contact)
            if ((is_robot_0 and is_furniture_1 and not is_gripper_0) or (is_robot_1 and is_furniture_0 and not is_gripper_1)):
                if has_impulse:
                    self.collision_metrics["robot_to_furniture"] += 1
                    step_has_furniture = True
                    count_furniture = True

            # Static objects: only check pose change (e.g. object knocked over / fallen); exclude gripper links
            if ((is_robot_0 and is_static_1 and not is_gripper_0) or (is_robot_1 and is_static_0 and not is_gripper_1)):
                static_name = name1 if is_static_1 else name0
                if self._static_object_has_significant_pose_change(static_name):
                    self.collision_metrics["robot_to_static_object"] += 1
                    step_has_static = True
                    count_static = True

            if (is_target_0 and is_static_1) or (is_target_1 and is_static_0):
                static_name = name1 if is_static_1 else name0
                if self._static_object_has_significant_pose_change(static_name):
                    self.collision_metrics["target_to_static_object"] += 1
                    step_has_target_static = True
                    count_target_static = True

            if count_furniture or count_static or count_target_static:
                for pt in contact.points:
                    impulse = float(np.linalg.norm(pt.impulse))
                    # Log furniture contacts by impulse; log static contacts regardless
                    if count_furniture and impulse > self.collision_impulse_threshold:
                        self.filtered_contacts_for_log.append({
                            "body0": name0,
                            "body1": name1,
                            "impulse": impulse,
                            "position": [float(x) for x in pt.position],
                        })
                    elif (count_static or count_target_static) and impulse > 0:
                        self.filtered_contacts_for_log.append({
                            "body0": name0,
                            "body1": name1,
                            "impulse": impulse,
                            "position": [float(x) for x in pt.position],
                        })

        if step_has_furniture:
            self.collision_metrics["robot_to_furniture_steps"] += 1
        if step_has_static:
            self.collision_metrics["robot_to_static_object_steps"] += 1
        if step_has_target_static:
            self.collision_metrics["target_to_static_object_steps"] += 1

        # Update previous-step poses for next iteration (step-to-step pose change detection)
        for entity in self.scene.get_all_actors():
            name = entity.get_name()
            if name in self.static_object_names:
                pose = entity.get_pose()
                self.static_object_pose_prev[name] = (
                    np.array(pose.p, dtype=np.float64),
                    np.array(pose.q, dtype=np.float64),
                )

    def get_collision_metrics(self):
        """Return a copy of current collision metrics dict."""
        return dict(self.collision_metrics)

    # =========================================================== Camera ===========================================================

    def load_camera(self, **kwags):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """

        self.cameras = Camera(
            bias=self.table_z_bias,
            random_head_camera_dis=self.random_head_camera_dis,
            **kwags,
        )
        self.add_extra_cameras() # extra cameras specific to task env
        self.cameras.load_camera(self.scene)
        self.scene.step()  # run a physical step
        self.scene.update_render()  # sync pose from SAPIEN to renderer
    
    def add_extra_cameras(self):
        pass

    # =========================================================== Basic APIs ===========================================================

    def add_prohibit_area(
        self,
        actor: Actor | sapien.Entity | sapien.Pose | list | np.ndarray,
        padding=0.01,
        area="table"
    ):
        if (isinstance(actor, sapien.Pose) or isinstance(actor, list) or isinstance(actor, np.ndarray)):
            actor_pose = transforms._toPose(actor)
            actor_data = {}
            scale = 1
        else:
            actor_pose = actor.get_pose()
            if isinstance(actor, Actor):
                actor_data = actor.config
            else:
                actor_data = {}
            scale = actor.scale
            
        origin_bounding_size = (np.array(actor_data.get("extents", [0.1, 0.1, 0.1])) * scale / 2)
        origin_bounding_pts = (np.array([
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]) * origin_bounding_size)

        if isinstance(padding,float) or isinstance(padding,int):
            padding = [padding, padding]

        actor_matrix = actor_pose.to_transformation_matrix()
        trans_bounding_pts = actor_matrix[:3, :3] @ origin_bounding_pts.T + actor_matrix[:3, 3].reshape(3, 1)
        x_min = np.min(trans_bounding_pts[0]) - padding[0]
        x_max = np.max(trans_bounding_pts[0]) + padding[0]
        y_min = np.min(trans_bounding_pts[1]) - padding[1]
        y_max = np.max(trans_bounding_pts[1]) + padding[1]
        # add_robot_visual_box(self, [x_min, y_min, actor_matrix[3, 3]])
        # add_robot_visual_box(self, [x_max, y_max, actor_matrix[3, 3]])
        self.prohibited_area[area].append([x_min, y_min, x_max, y_max])

    def choose_grasp_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id: list | float = None,
    ) -> list:
        """
        Test the grasp pose function.
        - actor: The actor to be grasped.
        - arm_tag: The arm to be used for grasping, either "left" or "right".
        - pre_dis: The distance in front of the grasp point, default is 0.1.
        """
        if not self.plan_success:
            return
        res_pre_top_down_pose = None
        res_top_down_pose = None
        res_id_top_down = None
        dis_top_down = 1e9
        res_pre_side_pose = None
        res_side_pose = None
        res_id_side = None
        dis_side = 1e9
        res_pre_pose = None
        res_pose = None
        res_id = None
        dis = 1e9

        pref_direction = self.robot.get_grasp_perfect_direction(arm_tag)

        def get_grasp_pose(pre_grasp_pose, pre_grasp_dis):
            grasp_pose = deepcopy(pre_grasp_pose)
            grasp_pose = np.array(grasp_pose)
            direction_mat = t3d.quaternions.quat2mat(grasp_pose[-4:])
            grasp_pose[:3] += [pre_grasp_dis, 0, 0] @ np.linalg.inv(direction_mat)
            grasp_pose = grasp_pose.tolist()
            return grasp_pose

        def check_pose(pre_pose, pose, arm_tag):
            if arm_tag == "left":
                plan_func = self.robot.left_plan_path
            else:
                plan_func = self.robot.right_plan_path
            pre_path = plan_func(pre_pose)
            if pre_path["status"] != "Success":
                return False
            pre_qpos = pre_path["position"][-1]
            return plan_func(pose)["status"] == "Success"

        if contact_point_id is not None:
            if type(contact_point_id) != list:
                contact_point_id = [contact_point_id]
            contact_point_id = [(i, None) for i in contact_point_id]
        else:
            contact_point_id = actor.iter_contact_points()
        for i, _ in contact_point_id:
            pre_pose = self.get_grasp_pose(actor, arm_tag, contact_point_id=i, pre_dis=pre_dis)
            if pre_pose is None:
                continue
            pose = get_grasp_pose(pre_pose, pre_dis - target_dis)
            now_dis_top_down = cal_quat_dis(
                pose[-4:],
                GRASP_DIRECTION_DIC[("top_down_little_left" if arm_tag == "right" else "top_down_little_right")],
            )
            now_dis_side = cal_quat_dis(pose[-4:], GRASP_DIRECTION_DIC[pref_direction])

            if res_pre_top_down_pose is None or now_dis_top_down < dis_top_down:
                res_pre_top_down_pose = pre_pose
                res_top_down_pose = pose
                res_id_top_down = i
                dis_top_down = now_dis_top_down

            if res_pre_side_pose is None or now_dis_side < dis_side:
                res_pre_side_pose = pre_pose
                res_side_pose = pose
                res_id_side = i
                dis_side = now_dis_side

            now_dis = 0.7 * now_dis_top_down + 0.3 * now_dis_side
            if res_pre_pose is None or now_dis < dis:
                res_pre_pose = pre_pose
                res_pose = pose
                res_id = i
                dis = now_dis
                
        if dis_top_down < 0.15:
            # print(f"choose_grasp_pose: selected contact_point_id={res_id_top_down} (top_down)")
            return res_pre_top_down_pose, res_top_down_pose
        if dis_side < 0.15:
            # print(f"choose_grasp_pose: selected contact_point_id={res_id_side} (side)")
            return res_pre_side_pose, res_side_pose
        # print(f"choose_grasp_pose: selected contact_point_id={res_id} (combined)")
        return res_pre_pose, res_pose

    def grasp_actor(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_grasp_dis=0.1,
        grasp_dis=0,
        gripper_pos=0.0,
        contact_point_id: list | float = None,
    ):
        if not self.plan_success:
            return None, []
        if self.need_plan == False:
            if pre_grasp_dis == grasp_dis:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]
            else:
                return arm_tag, [
                    Action(arm_tag, "move", target_pose=[0, 0, 0, 0, 0, 0, 0]),
                    Action(
                        arm_tag,
                        "move",
                        target_pose=[0, 0, 0, 0, 0, 0, 0],
                        constraint_pose=[1, 1, 1, 0, 0, 0],
                    ),
                    Action(arm_tag, "close", target_gripper_pos=gripper_pos),
                ]

        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=pre_grasp_dis,
            target_dis=grasp_dis,
            contact_point_id=contact_point_id,
        )

        if pre_grasp_pose is None:
            print("[ERROR] can't find a valid pre_grasp_pose")
            self.plan_success = False
            return None, []

        if pre_grasp_pose == grasp_pose:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]
        else:
            return arm_tag, [
                Action(arm_tag, "move", target_pose=pre_grasp_pose),
                Action(
                    arm_tag,
                    "move",
                    target_pose=grasp_pose,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                ),
                Action(arm_tag, "close", target_gripper_pos=gripper_pos),
            ]

    def get_place_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        target_pose: list | np.ndarray,
        constrain: Literal["free", "align", "auto"] = "auto",
        align_axis: list[np.ndarray] | np.ndarray | list = None,
        actor_axis: np.ndarray | list = [1, 0, 0],
        actor_axis_type: Literal["actor", "world"] = "actor",
        local_up_axis: np.ndarray | list | None = None,
        functional_point_id: int = None,
        pre_dis: float = 0.1,
        pre_dis_axis: Literal["grasp", "fp"] | np.ndarray | list = "grasp",
    ):

        if not self.plan_success:
            return [-1, -1, -1, -1, -1, -1, -1]

        actor_matrix = actor.get_pose().to_transformation_matrix()
        if functional_point_id is not None:
            place_start_pose = actor.get_functional_point(functional_point_id, "pose")
            z_transform = False
        else:
            place_start_pose = actor.get_pose()
            z_transform = True

        end_effector_pose = (self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose())

        if constrain == "auto":
            grasp_direct_vec = place_start_pose.p - end_effector_pose[:3]
            if np.abs(np.dot(grasp_direct_vec, [0, 0, 1])) <= 0.1:
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=grasp_direct_vec,
                    actor_axis_type="world",
                    align_axis=[1, 1, 0] if arm_tag == "left" else [-1, 1, 0],
                    z_transform=z_transform,
                    local_up_axis=local_up_axis,
                )
            else:
                camera_vec = transforms._toPose(end_effector_pose).to_transformation_matrix()[:3, 2]
                place_pose = get_place_pose(
                    place_start_pose,
                    target_pose,
                    constrain="align",
                    actor_axis=camera_vec,
                    actor_axis_type="world",
                    align_axis=[0, 1, 0],
                    z_transform=z_transform,
                    local_up_axis=local_up_axis,
                )
        else:
            place_pose = get_place_pose(
                place_start_pose,
                target_pose,
                constrain=constrain,
                actor_axis=actor_axis,
                actor_axis_type=actor_axis_type,
                align_axis=align_axis,
                z_transform=z_transform,
                local_up_axis=local_up_axis,
            )
        start2target = (transforms._toPose(place_pose).to_transformation_matrix()[:3, :3]
                        @ place_start_pose.to_transformation_matrix()[:3, :3].T)
        target_point = (start2target @ (actor_matrix[:3, 3] - place_start_pose.p).reshape(3, 1)).reshape(3) + np.array(
            place_pose[:3])

        ee_pose_matrix = t3d.quaternions.quat2mat(end_effector_pose[-4:])
        target_grasp_matrix = start2target @ ee_pose_matrix

        res_matrix = np.eye(4)
        res_matrix[:3, 3] = actor_matrix[:3, 3] - end_effector_pose[:3]
        res_matrix[:3, 3] = np.linalg.inv(ee_pose_matrix) @ res_matrix[:3, 3]
        target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)

        grasp_bias = target_grasp_matrix @ res_matrix[:3, 3]
        if pre_dis_axis == "grasp":
            target_dis_vec = target_grasp_matrix @ res_matrix[:3, 3]
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        else:
            target_pose_mat = transforms._toPose(target_pose).to_transformation_matrix()
            if pre_dis_axis == "fp":
                pre_dis_axis = [0.0, 0.0, 1.0]
            pre_dis_axis = np.array(pre_dis_axis)
            pre_dis_axis /= np.linalg.norm(pre_dis_axis)
            target_dis_vec = (target_pose_mat[:3, :3] @ np.array(pre_dis_axis).reshape(3, 1)).reshape(3)
            target_dis_vec /= np.linalg.norm(target_dis_vec)
        res_pose = (target_point - grasp_bias - pre_dis * target_dis_vec).tolist() + target_grasp_qpose.tolist()
        return res_pose

    def take_action(self, action, action_type:Literal['qpos', 'ee']='qpos'):  # action_type: qpos or ee
        if self.take_action_cnt == self.step_lim or self.eval_success:
            return

        eval_video_freq = 1  # fixed
        if (self.eval_video_path is not None and self.take_action_cnt % eval_video_freq == 0):
            self.eval_video_ffmpeg.stdin.write(self.now_obs["observation"]["demo_camera"]["rgb"].tobytes())

        self.take_action_cnt += 1
        print(f"step: \033[92m{self.take_action_cnt} / {self.step_lim}\033[0m", end="\r")

        self._update_render()
        if self.render_freq:
            self.viewer.render()

        actions = np.array([action])
        left_jointstate = self.robot.get_left_arm_jointState()
        right_jointstate = self.robot.get_right_arm_jointState()
        left_arm_dim = len(left_jointstate) - 1 if action_type == 'qpos' else 7
        right_arm_dim = len(right_jointstate) - 1 if action_type == 'qpos' else 7
        current_jointstate = np.array(left_jointstate + right_jointstate)

        left_arm_actions, left_gripper_actions, left_current_qpos, left_path = (
            [],
            [],
            [],
            [],
        )
        right_arm_actions, right_gripper_actions, right_current_qpos, right_path = (
            [],
            [],
            [],
            [],
        )

        left_arm_actions, left_gripper_actions = (
            actions[:, :left_arm_dim],
            actions[:, left_arm_dim],
        )
        right_arm_actions, right_gripper_actions = (
            actions[:, left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
            actions[:, left_arm_dim + right_arm_dim + 1],
        )
        left_current_gripper, right_current_gripper = (
            self.robot.get_left_gripper_val(),
            self.robot.get_right_gripper_val(),
        )

        left_gripper_path = np.hstack((left_current_gripper, left_gripper_actions))
        right_gripper_path = np.hstack((right_current_gripper, right_gripper_actions))

        if action_type == 'qpos':
            left_current_qpos, right_current_qpos = (
                current_jointstate[:left_arm_dim],
                current_jointstate[left_arm_dim + 1:left_arm_dim + right_arm_dim + 1],
            )
            left_path = np.vstack((left_current_qpos, left_arm_actions))
            right_path = np.vstack((right_current_qpos, right_arm_actions))

            # ========== TOPP ==========
            # TODO
            topp_left_flag, topp_right_flag = True, True

            try:
                times, left_pos, left_vel, acc, duration = (self.robot.left_mplib_planner.TOPP(left_path,
                                                                                            1 / 250,
                                                                                            verbose=True))
                left_result = dict()
                left_result["position"], left_result["velocity"] = left_pos, left_vel
                left_n_step = left_result["position"].shape[0]
            except Exception as e:
                # print("left arm TOPP error: ", e)
                topp_left_flag = False
                left_n_step = 50  # fixed

            if left_n_step == 0:
                topp_left_flag = False
                left_n_step = 50  # fixed

            try:
                times, right_pos, right_vel, acc, duration = (self.robot.right_mplib_planner.TOPP(right_path,
                                                                                                1 / 250,
                                                                                                verbose=True))
                right_result = dict()
                right_result["position"], right_result["velocity"] = right_pos, right_vel
                right_n_step = right_result["position"].shape[0]
            except Exception as e:
                # print("right arm TOPP error: ", e)
                topp_right_flag = False
                right_n_step = 50  # fixed

            if right_n_step == 0:
                topp_right_flag = False
                right_n_step = 50  # fixed
        
        elif action_type == 'ee':

            left_result = self.robot.left_plan_path(left_arm_actions[0])
            right_result = self.robot.right_plan_path(right_arm_actions[0])
            if left_result["status"] != "Success":
                left_n_step = 50
                topp_left_flag = False
                # print("left fail")
            else: 
                left_n_step = left_result["position"].shape[0]
                topp_left_flag = True
            
            if right_result["status"] != "Success":
                right_n_step = 50
                topp_right_flag = False
                # print("right fail")
            else:
                right_n_step = right_result["position"].shape[0]
                topp_right_flag = True

        # ========== Gripper ==========

        left_mod_num = left_n_step % len(left_gripper_actions)
        right_mod_num = right_n_step % len(right_gripper_actions)
        left_gripper_step = [0] + [
            left_n_step // len(left_gripper_actions) + (1 if i < left_mod_num else 0)
            for i in range(len(left_gripper_actions))
        ]
        right_gripper_step = [0] + [
            right_n_step // len(right_gripper_actions) + (1 if i < right_mod_num else 0)
            for i in range(len(right_gripper_actions))
        ]

        left_gripper = []
        for gripper_step in range(1, left_gripper_path.shape[0]):
            region_left_gripper = np.linspace(
                left_gripper_path[gripper_step - 1],
                left_gripper_path[gripper_step],
                left_gripper_step[gripper_step] + 1,
            )[1:]
            left_gripper = left_gripper + region_left_gripper.tolist()
        left_gripper = np.array(left_gripper)

        right_gripper = []
        for gripper_step in range(1, right_gripper_path.shape[0]):
            region_right_gripper = np.linspace(
                right_gripper_path[gripper_step - 1],
                right_gripper_path[gripper_step],
                right_gripper_step[gripper_step] + 1,
            )[1:]
            right_gripper = right_gripper + region_right_gripper.tolist()
        right_gripper = np.array(right_gripper)

        now_left_id, now_right_id = 0, 0

        # ========== Control Loop ==========
        while now_left_id < left_n_step or now_right_id < right_n_step:

            if (now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step):
                if topp_left_flag:
                    self.robot.set_arm_joints(
                        left_result["position"][now_left_id],
                        left_result["velocity"][now_left_id],
                        "left",
                    )
                self.robot.set_gripper(left_gripper[now_left_id], "left")

                now_left_id += 1

            if (now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step):
                if topp_right_flag:
                    self.robot.set_arm_joints(
                        right_result["position"][now_right_id],
                        right_result["velocity"][now_right_id],
                        "right",
                    )
                self.robot.set_gripper(right_gripper[now_right_id], "right")

                now_right_id += 1

            self.scene.step()
            self._update_render()

            if getattr(self, 'enable_collision_metrics', False) and hasattr(self, 'robot_link_names'):
                self.check_collisions()

            if self.check_success():
                self.eval_success = True
                self.get_obs() # update obs
                if (self.eval_video_path is not None):
                    self.eval_video_ffmpeg.stdin.write(self.now_obs["observation"]["head_camera"]["rgb"].tobytes())
                return

        self._update_render()
        if self.render_freq:  # UI
            self.viewer.render()
    
    # =========================================================== Extra Curobo Utils ===========================================================

    def update_world(self):
        """Updates CuRobo Collision World Model with new collision objects"""
        collision_dict = {"mesh": {}, "cuboid": {}}
        if self.collision_list:
            for info in self.collision_list:
                    actor = info["actor"]
                    collision_path = info["collision_path"]
                    if os.path.isdir(collision_path): # if actor is made from multiple obj files
                        name_prefix = actor.get_name()
                        if "link" in info:
                            if isinstance(info["link"], list):
                                pose = sapien.Pose()
                                pose.p = actor.get_link_pose(info["link"][0]).p
                                pose.q = actor.get_link_pose(info["link"][1]).q
                            else:
                                pose = actor.get_link_pose(info["link"])
                        elif "pose" in info:
                            pose = info["pose"]
                        else:
                            pose = actor.get_pose()
                        np_pose = np.concatenate([pose.p, pose.q]).tolist()
                        convex_collision_dict = self.collision_dict_from_convex_obj_dir(
                            collision_path,
                            pose=np_pose,
                            scale=actor.scale,
                            name_prefix = name_prefix,
                            files = info.get("files", None)
                        )
                        collision_dict["mesh"] = (
                            collision_dict["mesh"] | convex_collision_dict["mesh"]
                        )
                    else:
                        if "pose" in info:
                            pose = info["pose"]
                        else:
                            pose = actor.get_pose()
                        np_pose = np.concatenate([pose.p, pose.q]).tolist()
                        collision_dict["mesh"][f"{actor.get_name()}_{np_pose}_{self.seed}"] = {
                                "file_path": collision_path,
                                "pose": np_pose,
                                "scale": actor.scale,
                            }

        if self.cuboid_collision_list:
            for info in self.cuboid_collision_list:
                name = info["name"]
                dims = info["dims"]
                pose = info["pose"]
                collision_dict["cuboid"][f"{name}_{pose}_{self.seed}"] = {
                    "dims": dims,
                    "pose": pose,
                }
        self.robot.update_world(collision_dict)
    
    def collision_dict_from_convex_obj_dir(
        self,
        obj_dir: str | Path,
        *,
        name_prefix: str = "shelf_part",
        pose: tuple[float, float, float, float, float, float, float],  # [x,y,z,qw,qx,qy,qz]
        scale: tuple[float, float, float],  # e.g. (0.6, 0.8, 0.4)
        glob_pattern: str = "*.obj",
        files: list[str] = None,
        recursive: bool = False,
    ) -> dict:
        """
        Used to convert a directory of obj files into a dict of collision objects for curobo planner.
        Returns collision_dict in the form:
        collision_dict["mesh"][<name>] = {"file_path": ..., "pose": ..., "scale": ...}

        One entry per OBJ file (skips invalid/empty OBJs).
        """
        obj_dir = Path(obj_dir)
        if not obj_dir.exists() or not obj_dir.is_dir():
            raise FileNotFoundError(f"OBJ directory not found or not a directory: {obj_dir}")

        if files is not None:
            obj_files = []
            for file_name in files:
                p = obj_dir / file_name
                if p.is_file():
                    obj_files.append(p)
            obj_files = sorted(obj_files)
        else:
            it = obj_dir.rglob(glob_pattern) if recursive else obj_dir.glob(glob_pattern)
            obj_files = sorted([p for p in it if p.is_file()])

        if not obj_files:
            if files is not None:
                raise FileNotFoundError(
                    f"No requested OBJ files found in {obj_dir}. Requested files: {files}"
                )
            raise FileNotFoundError(
                f"No OBJ files found in {obj_dir} with pattern '{glob_pattern}' (recursive={recursive})"
            )

        collision_dict = {"mesh": {}}

        for i, p in enumerate(obj_files):
            # Validate OBJ so cuRobo/trimesh won't crash later
            try:
                m = trimesh.load(str(p), force="mesh", process=False)
            except Exception: # means the obj file is invalid
                continue

            if isinstance(m, trimesh.Scene):
                if len(m.geometry) == 0:
                    continue
                # concatenate ensures vertices/faces exist
                m = trimesh.util.concatenate(tuple(m.geometry.values()))

            if getattr(m, "vertices", None) is None or len(m.vertices) == 0:
                continue
            if getattr(m, "faces", None) is None or len(m.faces) == 0:
                continue

            part_name = f"{p}_{self.seed}"
            collision_dict["mesh"][part_name] = {
                "file_path": str(p),
                "pose": list(pose),
                "scale": list(scale),
            }
        
        if not collision_dict["mesh"]:
            raise ValueError(
                f"No valid mesh files were added from directory: {obj_dir} "
            )

        return collision_dict
        
    def attach_object(self, actor, file_path, arms_tag: str):
        """
        Attach a held object to the robot in Curobo Planning. Currently supports Actor or ArticulationActor.
        """
        pose = actor.get_pose()
        np_pose = np.concatenate([pose.p, pose.q]).tolist()
        object = {
            "name": actor.get_name(),
            "pose": np_pose,
            "file_path": file_path,
            "scale": actor.scale,
        }
        self.robot.attach_object(object, arms_tag=arms_tag)

    def detach_object(self, arms_tag: str):
        """
        Detach the attached objects from the robot in Curobo Planning.
        """
        self.robot.detach_object(arms_tag=arms_tag)
    
    def enable_obstacle(self, enable: bool, mesh_names: list[str] = [], obb_names: list[str] = []):
        self.robot.enable_obstacle(enable, mesh_names=mesh_names, obb_names=obb_names)

    def add_gripper_operating_area(self):
        # prohibit the area under the gripper start state so there are no initial collisions with obstacles
        if "table" not in self.prohibited_area:
            return
        x_half_width = 0.075
        ymax = -0.18
        ymin = -0.26
        self.prohibited_area["table"].append([-0.3-x_half_width, ymin, -0.3+x_half_width, ymax])
        self.prohibited_area["table"].append([0.3-x_half_width, ymin, 0.3+x_half_width, ymax])
    
    def add_operating_area(self, pose, width = 0.07, length = 0.28, direction = "forward"):
        if "table" not in self.prohibited_area:
            return
        # add a prohibited area in the space where the arm approaches a grasp or place. For horizontal movement.
        if direction == "forward": # from -y to +y
            xmin = pose[0] - width/2
            xmax = pose[0] + width/2
            ymin = pose[1] - length
            ymax = pose[1]
        elif direction == "right": # from -x to +x
            xmin = pose[0] - length
            xmax = pose[0]
            ymin = pose[1] - width/2
            ymax = pose[1] + width/2
        elif direction == "left": # from +x to -x
            xmin = pose[0]
            xmax = pose[0] + length
            ymin = pose[1] - width/2
            ymax = pose[1] + width/2
        self.prohibited_area["table"].append([xmin, ymin, xmax, ymax])
