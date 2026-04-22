import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS

from pathlib import Path
from typing import Optional, Sequence
import trimesh

try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.geom.types import WorldConfig, Mesh
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
            collision_cache: dict = {"mesh": 1, "obb": 1},
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        # "table": {
                        #     "dims": [0.7, 2, 0.04],  # x, y, z
                        #     "pose": [
                        #         self.robot_origion_pose.p[1],
                        #         0.0,
                        #         0.74 - self.robot_origion_pose.p[2],
                        #         1,
                        #         0,
                        #         0,
                        #         0.0,
                        #     ],  # x, y, z, qw, qx, qy, qz
                        # },
                    },
                    "mesh": {}
                }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=8,
                collision_cache=collision_cache,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()
            
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=8,
                num_graph_seeds=1,
                collision_cache=collision_cache,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):  
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

            # ------------------------------------------
            if result.success.item() == False:
                print("[Error]: CuroboPlanner plan_path failed:", result.status)
            # ------------------------------------------

            # output
            res_result = dict()
            if result.success.item() == False:
                res_result["status"] = "Fail"
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else: # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except Exception as e:
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
        
        def update_world(self, collision_dict, arms_tag):
            """Update CuRobo Collision World Model with new collision objects"""

            collision_dict = self.collision_dict_world_to_arm_base(collision_dict, arms_tag)
            world_config = WorldConfig.from_dict(collision_dict)

            for gen in [self.motion_gen, self.motion_gen_batch]:
                gen.clear_world_cache()
                gen.update_world(world_config)
            
            # world_model = self.motion_gen.world_coll_checker.world_model
            # scene = self.visualize_world_config(world_model)
            # scene.show()
            
        def world_to_arm_base_pose(self, world_pose, arms_tag=None):
            """
            Convert a pose from world frame to arm base frame.

            Args:
                world_pose: either
                    - object with .p and .q
                    - or iterable [x, y, z, qw, qx, qy, qz]
                arms_tag: "left" or "right" (required for aloha-agilex patch)

            Returns:
                (p, q) where:
                    p: np.ndarray (3,)
                    q: np.ndarray (4,) quaternion
            """
            import numpy as np
            import transforms3d as t3d

            # --- Extract world pose ---
            if hasattr(world_pose, "p") and hasattr(world_pose, "q"):
                world_target_pose = np.concatenate([
                    np.array(world_pose.p),
                    np.array(world_pose.q),
                ])
            else:
                world_target_pose = np.array(world_pose)

            # --- World base pose ---
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])

            # --- World → robot base ---
            target_pose_p, target_pose_q = self._trans_from_world_to_base(
                world_base_pose,
                world_target_pose,
            )

            # --- Frame bias / aloha patch ---
            if "aloha-agilex" not in self.yml_path:
                target_pose_p = target_pose_p + np.array(self.frame_bias)

            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(
                    target_pose_p,
                    t3d.quaternions.quat2mat(target_pose_q),
                    [1, 1, 1],
                )

                T_bias = t3d.affines.compose(
                    self.frame_bias,
                    np.eye(3),
                    [1, 1, 1],
                )

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target

                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            return target_pose_p, target_pose_q

        def collision_dict_world_to_arm_base(self, collision_dict: dict, arms_tag: str | None = None) -> dict:
            """
            Returns a copy of collision_dict where every obstacle pose is converted from world frame
            to arm base frame using self.world_to_arm_base_pose(...).

            Expects each obstacle to have a 'pose' field as [x,y,z,qw,qx,qy,qz].
            """
            import copy
            import numpy as np

            out = copy.deepcopy(collision_dict)

            # helper: convert a flat pose list -> flat pose list
            def _convert_pose_list(pose_list):
                p, q = self.world_to_arm_base_pose(pose_list, arms_tag=arms_tag)
                return list(np.asarray(p).tolist()) + list(np.asarray(q).tolist())

            # meshes
            if "mesh" in out and out["mesh"]:
                for name, md in out["mesh"].items():
                    if "pose" in md and md["pose"] is not None:
                        md["pose"] = _convert_pose_list(md["pose"])

            # cuboids
            if "cuboid" in out and out["cuboid"]:
                for name, cd in out["cuboid"].items():
                    if "pose" in cd and cd["pose"] is not None:
                        cd["pose"] = _convert_pose_list(cd["pose"])

            # add other primitives here if you use them (sphere/capsule/cylinder/etc.)

            return out
        
        def visualize_world_config(self, world_model):
            """Visualize CuRobo Collision World Model: motion_gen.world_coll_checker.world_model"""
            from scipy.spatial.transform import Rotation

            scene = trimesh.Scene()
            
            # Add all cuboids
            for cuboid in world_model.cuboid:
                # Create box
                box = trimesh.creation.box(extents=cuboid.dims)
                
                # Create transformation matrix
                pose = cuboid.pose
                transform = np.eye(4)
                
                # Rotation from quaternion [qw, qx, qy, qz]
                quat = [pose[3], pose[4], pose[5], pose[6]]
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [qx,qy,qz,qw]
                transform[:3, :3] = rot.as_matrix()
                
                # Translation
                transform[:3, 3] = [pose[0], pose[1], pose[2]]
                
                # Apply transform
                box.apply_transform(transform)
                
                # Set color (red for collision objects)
                box.visual.face_colors = [255, 0, 0, 100]  # Red, semi-transparent
                
                # Add to scene
                scene.add_geometry(box, node_name=cuboid.name)
            
            # Add all meshes
            for mesh_obj in world_model.mesh:
                try:
                    # Load mesh from file
                    if mesh_obj.file_path:
                        mesh = trimesh.load(mesh_obj.file_path, force='mesh')
                        
                        # Apply scale
                        mesh.apply_scale(mesh_obj.scale)
                        
                        # Apply pose
                        pose = mesh_obj.pose
                        transform = np.eye(4)
                        
                        quat = [pose[3], pose[4], pose[5], pose[6]]
                        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
                        transform[:3, :3] = rot.as_matrix()
                        transform[:3, 3] = [pose[0], pose[1], pose[2]]
                        
                        mesh.apply_transform(transform)

                        # Set color (blue for meshes)
                        mesh.visual.face_colors = [0, 0, 255, 100]
                        
                        scene.add_geometry(mesh, node_name=mesh_obj.name)
                        
                except Exception as e:
                    print(f"Could not load mesh {mesh_obj.name}: {e}")
                
            return scene

        def enable_obstacle(self, enable: bool, mesh_names: list[str] = [], obb_names: list[str] = []):
            for name in mesh_names:
                self.motion_gen.world_coll_checker.enable_mesh(enable,name=name)
                self.motion_gen_batch.world_coll_checker.enable_mesh(enable,name=name)
            for name in obb_names:
                self.motion_gen.world_coll_checker.enable_obb(enable, name=name)
                self.motion_gen_batch.world_coll_checker.enable_obb(enable, name=name)

        def attach_object(self, object: dict, curr_joint_pos: list, arms_tag: str):
            """
            Attach an object to the robot in Curobo Planning.
            Args:
                object: dict, the object to attach
                curr_joint_pos: list, the current joint positions
                arms_tag: str, the arm tag
            Returns:
                None
            """
            p,q = self.world_to_arm_base_pose(object["pose"], arms_tag=arms_tag) # convert object pose to arm base frame
            pose = np.concatenate([p, q]).tolist()
            # -----------------------------------------------
            obstacle = Mesh(
                name=object["name"],
                pose=pose,
                file_path=object["file_path"],
                scale=object["scale"],
            ) 
            # -----------------------------------------------
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]
            joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            # -----------------------------------------------

            for mg in [self.motion_gen, self.motion_gen_batch]:
                ok = mg.attach_external_objects_to_robot(
                    joint_state=joint_states,
                    external_objects=[obstacle],
                    link_name="attached_object",
                    surface_sphere_radius=0.005,  # start here; decrease for tighter fit
                    voxelize_method="ray",
                )
                assert ok
        
        def detach_object(self):
            for mg in [self.motion_gen, self.motion_gen_batch]:
                mg.detach_object_from_robot(link_name="attached_object",)

        def visualize_attached_objects(self, curr_joint_pos: list):
            world_model = self.motion_gen.world_coll_checker.world_model
            scene = self.visualize_world_config(world_model)
            # ---- Add attached object spheres ----
            kc = self.motion_gen.robot_cfg.kinematics.kinematics_config
            ss = kc.get_link_spheres("attached_object")

            valid_idx = (ss[:,3] > 0).nonzero().flatten()

            # get EE pose for correct transform
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]
            # -----------------------------------------------
            joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )

            ks = self.motion_gen.compute_kinematics(joint_states)
            w_T_ee = ks.ee_pose
            from curobo.types.math import Pose

            for i in valid_idx.tolist():
                x, y, z, r = ss[i].tolist()

                # sphere in EE frame
                ee_T_sphere = Pose.from_list([x, y, z, 0, 0, 0, 1])
                w_T_sphere = w_T_ee.multiply(ee_T_sphere)
                center = w_T_sphere.position.squeeze().cpu().numpy()

                # create trimesh sphere
                sphere_mesh = trimesh.creation.icosphere(radius=r, subdivisions=2)
                sphere_mesh.apply_translation(center)

                # green for attached object
                sphere_mesh.visual.face_colors = [0, 255, 0, 180]

                scene.add_geometry(sphere_mesh)
            
            scene.show()

    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
