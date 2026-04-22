from bench_envs.kitchens._kitchens_base_task import KitchenS_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_plate_in_sink_ks(KitchenS_base_task):

    # Fully scripted: top-down rim pinch, then drop over sink. Plate contacts
    # on model_data0 encode top-down rim orientations but grasp_actor's
    # chosen pose is frequently IK-unreachable for the aloha arm; bypassing
    # it mirrors what pick_apple_from_sink_ks does.
    TOP_DOWN_Q = [-0.5, 0.5, -0.5, -0.5]
    TCP_OFFSET = 0.12

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name()}

    def load_actors(self):
        sink_p = self.sink.get_pose().p
        self.arm_tag = ArmTag("right" if sink_p[0] > 0 else "left")
        side_sign = 1 if self.arm_tag == "right" else -1

        # Plate spawns on the same side as the sink/arm. Plate center offset
        # at least 0.18 so the near-side rim (at plate.x - 0.085 * side_sign)
        # sits comfortably inside the arm workspace. y narrowed to the
        # empirically reachable top-down envelope (successful seeds had
        # plate.y ∈ [-0.14, -0.09]).
        x_range = [0.18, 0.28] if side_sign > 0 else [-0.28, -0.18]

        rand_pos = self.rand_pose_on_counter(
            xlim=x_range,
            ylim=[-0.15, -0.08],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
            obj_padding=0.10,
        )

        self.plate_id = 0
        self.target_obj = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="003_plate",
            convex=True,
            model_id=self.plate_id,
        )
        self.target_obj.set_mass(0.1)

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")

    def play_once(self):
        arm_tag = self.arm_tag
        side_sign = 1 if arm_tag == "right" else -1

        # Counter cuboid must be out of the Curobo world so the wrist can
        # dip below rim z when placing plate into the sink basin.
        self.enable_table(enable=False)

        self.move(self.open_gripper(arm_tag, pos=1.0))

        plate_p = self.target_obj.get_pose().p
        # Rim point nearest the robot-arm base: inner side of plate along x.
        rim_x = float(plate_p[0]) - 0.085 * side_sign
        rim_y = float(plate_p[1])
        rim_z_top = float(plate_p[2]) + 0.012  # rim sits ~1.2 cm above base

        hover_tcp_z = rim_z_top + 0.10
        hover_pose = [rim_x, rim_y, hover_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, hover_pose))

        grasp_tcp_z = rim_z_top - 0.002
        grasp_pose = [rim_x, rim_y, grasp_tcp_z + self.TCP_OFFSET] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, grasp_pose))

        self.move(self.close_gripper(arm_tag, pos=0.0))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.attach_object(
            self.target_obj,
            f"{os.environ['ROBOTWIN_ROOT']}/assets/objects/003_plate/collision/base{self.plate_id}.glb",
            str(arm_tag),
        )

        sink_p = self.sink.get_pose().p
        drop_tcp_z = float(sink_p[2]) + 0.14
        drop_pose = [
            float(sink_p[0]) - 0.085 * side_sign,  # align plate rim over sink center
            float(sink_p[1]),
            drop_tcp_z + self.TCP_OFFSET,
        ] + self.TOP_DOWN_Q
        self.move(self.move_to_pose(arm_tag, drop_pose))

        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        sink_p = self.sink.get_pose().p
        sg = self.kitchens_info["sink_geom"]
        tp = self.target_obj.get_pose().p
        in_xy = abs(tp[0] - sink_p[0]) < sg["hole_hx"] and abs(tp[1] - sink_p[1]) < sg["hole_hy"]
        below_rim = tp[2] < sink_p[2] + 0.01
        return in_xy and below_rim and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open()
