from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.utils import sapien_utils
import gymnasium as gym
import numpy as np
import torch
import sapien
from typing import Union
from envs.utils import convert_observation_to_space, flatten_space


class MultiRobotWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = flatten_space(env.action_space)
        self.single_action_space = flatten_space(env.single_action_space)

    def action(self, action):
        ac = {}
        counter = 0
        for k, v in self.env.action_space.items():
            ac[k] = action[..., counter : counter + v.shape[-1]]
            counter += v.shape[-1]
        return ac


@register_agent()
class PandaWristCamPegCustom(PandaWristCam):
    """Panda arm robot with the real sense camera attached to gripper"""

    uid = "panda_wristcam_custom"

    @property
    def _sensor_configs(self):
        pose2 = sapien_utils.look_at([0.0, 0.0, 0.0], [0.0, 0.0, 0.4])
        pose3 = sapien_utils.look_at([0.0, 0.0, 0.0], [1.0, 0.0, 0.3])
        q = pose3.q.squeeze().tolist() if isinstance(pose3.q, torch.Tensor) else pose3.q
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=q),
                width=128,
                height=128,
                fov=1.2 * np.pi / 2,
                near=0.01,
                far=10,
                mount=self.robot.links_map["camera_link"],
            )
        ]


class DEMO3_BaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("dense", "sparse", "semi_sparse")

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda_wristcam_custom"]
    agent: Union[PandaWristCam, PandaWristCamPegCustom]

    def compute_stage_indicator(self):
        raise NotImplementedError()

    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            eval_info = self.evaluate(**kwargs)
            return float(eval_info["success"])
        elif self._reward_mode == "dense":
            return self.compute_dense_reward(**kwargs)
        elif self._reward_mode == "semi_sparse":
            # reward build from stage indicators
            return self.compute_semi_sparse_reward(**kwargs)
        else:
            raise NotImplementedError(self._reward_mode)

    def compute_semi_sparse_reward(self, **kwargs):
        stage_indicators = self.compute_stage_indicator()
        eval_info = self.evaluate()
        return sum(stage_indicators.values()) + eval_info["success"].float()

    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose_ext = sapien_utils.look_at(
            eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35]
        )  # NOTE: Same as render camera
        pose_base = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose_base,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "ext_camera",
                pose=pose_ext,
                width=128,
                height=128,
                fov=1,
                near=0.01,
                far=100,
            ),
        ]


############################################
# Pick And Place
############################################

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv


@register_env("PickAndPlace_DEMO3", max_episode_steps=100)
class PickAndPlace_DEMO3(DEMO3_BaseEnv, PickCubeEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, robot_uids="panda_wristcam", **kwargs)

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "is_grasped": (eval_info["is_grasped"]).float(),
            "is_obj_placed": (eval_info["is_obj_placed"]).float(),
        }


############################################
# Stack Cube
############################################

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv


@register_env("StackCube_DEMO3", max_episode_steps=100)
class StackCube_DEMO3(DEMO3_BaseEnv, StackCubeEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, robot_uids="panda_wristcam", **kwargs)

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "is_grasped": (
                torch.logical_or(eval_info["is_cubeA_grasped"], eval_info["success"])
            ).float(),  # allow releasing the cube when stacked
            "is_cube_A_placed": (
                torch.logical_or(eval_info["is_cubeA_on_cubeB"], eval_info["success"])
            ).float(),
        }


############################################
# Peg Insertion
############################################

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv


@register_env("PegInsertionSide_DEMO3", max_episode_steps=100)
class PegInsertionSide_DEMO3(DEMO3_BaseEnv, PegInsertionSideEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda_wristcam_custom"]
    agent: Union[PandaWristCam, PandaWristCamPegCustom]

    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, robot_uids="panda_wristcam_custom", **kwargs)

    def is_peg_pre_inserted(self):
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_head_wrt_goal_yz_dist = torch.linalg.norm(
            peg_head_wrt_goal.p[:, 1:], axis=1
        )
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        # stage 3 passes if peg is correctly oriented in order to insert into hole easily
        pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
            peg_wrt_goal_yz_dist < 0.01
        )
        return pre_inserted

    def compute_stage_indicator(self):
        success = self.evaluate()["success"]
        stage_1 = torch.logical_or(
            self.agent.is_grasping(self.peg, max_angle=20), success
        )
        stage_2 = torch.logical_or(self.is_peg_pre_inserted(), success)
        return {
            "is_correctly_grasped": torch.logical_or(
                stage_1, stage_2
            ).float(),  # do this to allow releasing the peg when inserted
            "is_peg_pre_inserted": stage_2.float(),
        }

    @property
    def _default_sensor_configs(self):
        # Define all the cameras needed for the environment
        pose_ext = sapien_utils.look_at(
            [0.5, -0.5, 0.8], [0.05, -0.1, 0.4]
        )  # NOTE: Same as render camera
        pose_base = sapien_utils.look_at([0, -0.4, 0.2], [0, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose_base,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            ),
            CameraConfig(
                "ext_camera",
                pose=pose_ext,
                width=128,
                height=128,
                fov=1,
                near=0.01,
                far=100,
            ),
        ]


############################################
# Lift Peg Upright
############################################

from mani_skill.envs.tasks.tabletop.lift_peg_upright import LiftPegUprightEnv
from mani_skill.utils.geometry import rotation_conversions


@register_env("LiftPegUpright_DEMO3", max_episode_steps=100)
class LiftPegUpright_DEMO3(DEMO3_BaseEnv, LiftPegUprightEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, robot_uids="panda_wristcam", **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ]
        )
        qpos = (
            self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
            + qpos
        )
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_half_turn = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.52
        )  # 0.08 radians of difference permitted
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        return {
            "is_peg_grasped": self.agent.is_grasping(self.peg),
            "is_peg_upright": is_peg_half_turn,
            "success": is_peg_upright & close_to_table,
        }

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(eval_info["is_peg_grasped"], eval_info["success"])
            ).float(),  # allow releasing the cube when stacked
            "stage_2": (
                torch.logical_or(eval_info["is_peg_upright"], eval_info["success"])
            ).float(),
        }


############################################
# Two Robot PickCube
############################################

from mani_skill.envs.tasks.tabletop.two_robot_pick_cube import TwoRobotPickCube
from mani_skill.utils.geometry import rotation_conversions


@register_env("TwoRobotPickCube_DEMO3", max_episode_steps=100)
class TwoRobotPickCube_DEMO3(DEMO3_BaseEnv, TwoRobotPickCube):
    def __init__(self, *args, **kwargs):
        self.n_stages = 4
        super().__init__(*args, **kwargs)

    def evaluate(self):
        # stage 1 passes if cube is near a sub-goal
        cube_at_other_side = self.cube.pose.p[:, 1] >= 0.0

        # stage 2 passes if cube is grasped by right arm
        is_grasped = self.right_agent.is_grasping(self.cube)

        # stage 3 passes if cube is in goal area
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )

        is_right_arm_static = self.right_agent.is_static(0.2)

        return {
            "is_cube_reachable": cube_at_other_side,
            "is_cube_grasped": is_grasped,
            "is_obj_placed": is_obj_placed,
            "is_right_arm_static": is_right_arm_static,
            "success": torch.logical_and(is_obj_placed, is_right_arm_static),
        }

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(eval_info["is_cube_reachable"], eval_info["success"])
            ).float(),  # allow releasing the cube when stacked
            "stage_2": (
                torch.logical_or(eval_info["is_cube_grasped"], eval_info["success"])
            ).float(),
            "stage_3": (
                torch.logical_or(eval_info["is_obj_placed"], eval_info["success"])
            ).float(),
        }


############################################
# Two Robot StackCube
############################################

from mani_skill.envs.tasks.tabletop.two_robot_stack_cube import TwoRobotStackCube


@register_env("TwoRobotStackCube_DEMO3", max_episode_steps=100)
class TwoRobotStackCube_DEMO3(DEMO3_BaseEnv, TwoRobotStackCube):
    def __init__(self, *args, **kwargs):
        self.n_stages = 4
        super().__init__(*args, **kwargs)

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(eval_info["cubeB_placed"], eval_info["success"])
            ).float(),  # allow releasing the cube when stacked
            "stage_2": (
                torch.logical_or(eval_info["is_cubeA_grasped"], eval_info["success"])
            ).float(),
            "stage_3": (
                torch.logical_or(eval_info["is_cubeA_on_cubeB"], eval_info["success"])
            ).float(),
        }


############################################
# Poke Cube
############################################

from mani_skill.envs.tasks.tabletop.poke_cube import PokeCubeEnv
from mani_skill.utils.geometry import rotation_conversions


@register_env("PokeCube_DEMO3", max_episode_steps=100)
class PokeCube_DEMO3(DEMO3_BaseEnv, PokeCubeEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, robot_uids="panda_wristcam_custom", **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ]
        )
        qpos = (
            self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos)))
            + qpos
        )
        qpos[:, -2:] = 0.04
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(eval_info["is_peg_grasped"], eval_info["success"])
            ).float(),  # allow releasing the cube when stacked
            "stage_2": (
                torch.logical_or(
                    eval_info["head_to_cube_dist"] <= (self.cube_half_size + 0.03),
                    eval_info["success"],
                )
            ).float(),
        }


############################################
# Humanoid Place Apple
############################################

from mani_skill.envs.tasks.humanoid import UnitreeG1PlaceAppleInBowlEnv


@register_env("HumanoidPlaceApple_DEMO3", max_episode_steps=100)
class HumanoidPlaceApple_DEMO3(DEMO3_BaseEnv, UnitreeG1PlaceAppleInBowlEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, **kwargs)

    def evaluate(self):
        is_obj_placed_xy = (
            torch.linalg.norm(
                self.bowl.pose.p[:, :-1] - self.apple.pose.p[:, :-1], axis=1
            )
            <= 0.1
        )
        is_obj_placed = (
            torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1) <= 0.05
        )
        hand_outside_bowl = (
            self.agent.right_tcp.pose.p[:, 2] > self.bowl.pose.p[:, 2] + 0.125
        )
        is_grasped = self.agent.right_hand_is_grasping(self.apple, max_angle=110)
        return {
            "success": is_obj_placed & hand_outside_bowl,
            "hand_outside_bowl": hand_outside_bowl,
            "is_obj_placed": is_obj_placed,
            "is_obj_placed_xy": is_obj_placed_xy,
            "is_grasped": is_grasped,
        }

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(eval_info["is_grasped"], eval_info["is_obj_placed_xy"])
            ).float(),  # allow releasing the cube when stacked
            "stage_2": eval_info["is_obj_placed_xy"].float(),
        }

    @property
    def _default_sensor_configs(self):
        return CameraConfig(
            "base_camera",
            sapien.Pose(
                [0.279123, 0.303438, 1.34794], [0.252428, 0.396735, 0.114442, -0.875091]
            ),
            128,
            128,
            np.pi / 2,
            0.01,
            100,
        )


############################################
# Humanoid Transport Box
############################################

from mani_skill.envs.tasks.humanoid import TransportBoxEnv


@register_env("HumanoidTransportBox_DEMO3", max_episode_steps=100)
class TransportBox_DEMO3(DEMO3_BaseEnv, TransportBoxEnv):
    def __init__(self, *args, **kwargs):
        self.n_stages = 3
        super().__init__(*args, **kwargs)

    def evaluate(self):
        # left_hand_grasped_box = self.agent.left_hand_is_grasping(self.box, max_angle=110)
        # right_hand_grasped_box = self.agent.right_hand_is_grasping(self.box, max_angle=110)
        l_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        r_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        left_hand_hit_box = l_contact_forces > 10
        right_hand_hit_box = r_contact_forces > 10
        # is grasping the box if both hands contact the box and the tcp of the hands are below the grasp points on the box.
        box_grasped = (
            left_hand_hit_box
            & right_hand_hit_box
            & (
                self.agent.right_tcp.pose.p[:, 2]
                < self.box_right_grasp_point.p[:, 2] + 0.04
            )
            & (
                self.agent.left_tcp.pose.p[:, 2]
                < self.box_left_grasp_point.p[:, 2] + 0.04
            )
        )

        # simply requires box to be resting somewhere on the correct table
        box_at_correct_table_z = (0.751 > self.box.pose.p[:, 2]) & (
            self.box.pose.p[:, 2] > 0.750
        )
        box_at_correct_table_xy = (
            (0.78 > self.box.pose.p[:, 0])
            & (self.box.pose.p[:, 0] > -0.78)
            & (1.0 > self.box.pose.p[:, 1])
            & (self.box.pose.p[:, 1] > 0.0)
        )
        # box_at_correct_table = torch.linalg.norm(self.box.pose.p - torch.tensor([0, 0.66, 0.731], device=self.device), dim=1) < 0.05
        box_at_correct_table = box_at_correct_table_z & box_at_correct_table_xy

        facing_table_with_box = (-1.7 < self.agent.robot.qpos[:, 0]) & (
            self.agent.robot.qpos[:, 0] < -1.4
        )  # in this range the robot is probably facing the box on the left table.
        return {
            "success": ~box_grasped & box_at_correct_table,
            "left_hand_hit_box": l_contact_forces > 0,
            "right_hand_hit_box": r_contact_forces > 0,
            "box_grasped": box_grasped,
            "box_at_correct_table_xy": box_at_correct_table_xy,
            "facing_table_with_box": facing_table_with_box,
        }

    def compute_stage_indicator(self):
        eval_info = self.evaluate()
        return {
            "stage_1": (
                torch.logical_or(
                    eval_info["box_grasped"], eval_info["box_at_correct_table_xy"]
                )
            ).float(),  # allow releasing the cube when stacked
            "stage_2": (
                torch.logical_or(
                    eval_info["box_at_correct_table_xy"], eval_info["success"]
                )
            ).float(),
        }

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]
