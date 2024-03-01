from scalingup.utils.core import (
    Pose,
    PointCloud,
    DoNothingAction,
    Env,
    Action,
    EndEffectorAction,
    normal_to_forward_quat,
    get_best_orn_for_gripper,
)
import numpy as np
import logging
from transforms3d import affines, euler, quaternions



class PlaceAction():

    def __init__(self) -> None:
        self.attempt_count = 0
        self.ee_action_sequence = None

    def point_cloud_to_place_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
    ):
        obs = env.obs
        state = obs.state
        numpy_random = env.policy_numpy_random

        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        filter_out_mask = normals[:, 2] < env.config.pointing_up_normal_threshold

        xyz_pts = link_point_cloud.xyz_pts[~filter_out_mask]
        normals = normals[~filter_out_mask]
        candidate: Action = DoNothingAction()
        # link_pose = obs.state.get_pose(key=self.link_path)
        # TODO sampling can be biased towards points that belong to larger surface areas
        num_candidates = (~filter_out_mask).sum()
        if num_candidates == 0:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"no candidates for {self.link_path}",
            )
        logging.debug(f"PlaceOnLinkAction has {num_candidates} candidates")
        for attempt, idx in enumerate(
            numpy_random.choice(
                num_candidates, size=env.config.num_action_candidates, replace=True
            )
        ):
            position = xyz_pts[idx].copy()
            # compute base orientation, randomize along Z-axis
            normal = normals[idx].copy()
            normal /= np.linalg.norm(normal)
            try:
                base_orientation = quaternions.qmult(
                    normal_to_forward_quat(normal),
                    euler.euler2quat(-np.pi, 0, np.pi),
                )
            except np.linalg.LinAlgError as e:
                logging.warning(e)
                base_orientation = euler.euler2quat(0, 0, 0)
            z_angle = numpy_random.uniform(np.pi, -np.pi)
            z_orn = euler.euler2quat(0, 0, z_angle)
            base_orientation = quaternions.qmult(base_orientation, z_orn)

            # compute grasp pose relative to object
            place_pose = Pose(
                position=position
                + normal
                * numpy_random.uniform(
                    env.config.place_height_min, env.config.place_height_max
                ),
                orientation=base_orientation,
            )#.transform(np.linalg.inv(link_pose.matrix))
            # NOTE: if too low, then gripper + object can also bump and move the
            # object it's placing on top of, like the previously stacked blocks or
            # drawer
            preplace_dist = numpy_random.uniform(
                env.config.preplace_dist_min, env.config.preplace_dist_max
            )
            # candidate = PlaceOnLinkPoseAction(
            #     link_path=self.link_path,
            #     pose=place_pose,
            #     preplace_dist=preplace_dist,
            # )
            rotmat = quaternions.quat2mat(place_pose.orientation)
            place_direction = rotmat @ np.array([0, 0, 1])
            preplace_pos = place_pose.position + place_direction * preplace_dist
            orn = get_best_orn_for_gripper(
                reference_orn=euler.euler2quat(0, 0, 0),
                query_orn=place_pose.orientation,
            )
            try:
                self.check_feasibility(env=env, pose=place_pose, orn=orn, preplace_pos=preplace_pos)
                logging.info(f"[{attempt}] preplace_dist: {preplace_dist}")
                break
            except Action.InfeasibleAction as e:
                logging.debug(e)
                place_pose = None
                continue
        if place_pose is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"all candidates failed: {self}",
            )
        
        ee_action_sequence =  [
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=preplace_pos,
                end_effector_orientation=orn,
                allow_contact=False,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=place_pose.position,
                end_effector_orientation=orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=place_pose.position,
                end_effector_orientation=orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=preplace_pos,
                end_effector_orientation=orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
        ]
        self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute_place(self, env, info={"log": "", "subtrajectories": []},):
        for i, ee_action in enumerate(self.ee_action_sequence):
            ee_action.execute_bebs(env, info=info)
        
    def check_feasibility(self, env: Env, pose: Pose, orn, preplace_pos):
        # first check for kinematic reachability
        result1 = env.robot.inverse_kinematics(
            Pose(
                position=preplace_pos,
                orientation=orn,
            )
        )
        result2 = env.robot.inverse_kinematics(
            Pose(
                position=pose.position,
                orientation=orn,
            )
        )
        if result1 is None or result2 is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__, message="IK Failed"
            )
    

