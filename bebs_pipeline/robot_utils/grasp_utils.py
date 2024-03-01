from scalingup.utils.core import (
    Pose,
    PointCloud,
    Env,
    Action,
    EndEffectorAction,
    normal_to_forward_quat,
    get_best_orn_for_gripper,
)
import numpy as np
import logging
from transforms3d import affines, euler, quaternions

top_down_grasp_bias = 1e-4
griper_closed_threshold = 0.02


class GraspAction():

    def __init__(self) -> None:
        self.attempt_count = 0
        self.ee_action_sequence = None

    def point_cloud_to_grasp_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
        with_backup=True, pushin_more=True
    ):
        state = env.obs.state
        numpy_random = env.policy_numpy_random
        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        pointing_down = normals[:, 2] < 0.0
        p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones point up more often
            p = np.exp(normals[:, 2] * top_down_grasp_bias)
        p /= p.sum()

        errors = []
        candidate_indices = numpy_random.choice(
            len(link_point_cloud),
            size=min(env.config.num_action_candidates, len(link_point_cloud)),
            p=p,
            replace=True,
        )
        for i, idx in enumerate(candidate_indices):
            position = link_point_cloud.xyz_pts[idx].copy()
            grasp_to_ee = state.end_effector_pose.position - position
            grasp_to_ee /= np.linalg.norm(grasp_to_ee)
            if pointing_down.all():
                # disallow bottom up grasps, so using point to ee
                # as normal, with some random exploration
                grasp_to_ee += numpy_random.randn(3) * 0.1
                grasp_to_ee /= np.linalg.norm(grasp_to_ee)
                normal = grasp_to_ee
            else:
                normal = normals[idx]

            # orient normal towards end effector
            if np.dot(grasp_to_ee, normal) < 0:
                normal *= -1.0
            # compute base orientation, randomize along Z-axis
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
            pregrasp_distance = numpy_random.uniform(0.1, 0.2)
            if pushin_more:
                # prioritize grasps that push in more, if possible, but slowly
                # back off to prevent all grasps colliding.
                pushin_distance = (len(candidate_indices) - self.attempt_count) / len(
                    candidate_indices
                ) * (
                    env.config.max_pushin_dist - env.config.min_pushin_dist
                ) + env.config.min_pushin_dist
            else:
                # prioritize grasps that push in less. Useful for "pushing"
                pushin_distance = (
                    self.attempt_count
                    / len(candidate_indices)
                    * (env.config.max_pushin_dist - env.config.min_pushin_dist)
                    + env.config.min_pushin_dist
                )
            # compute grasp pose relative to object
            # link_pose = state.get_pose(key='block/|block/block')
            grasp_pose = Pose(
                position=position - normal * pushin_distance,
                orientation=base_orientation,
            )#.transform(np.linalg.inv(link_pose.matrix))
            # candidate = GraspLinkPoseAction(
            #     link_path='block/|block/block',
            #     pose=grasp_pose,
            #     with_backup=with_backup,
            #     backup_distance=pushin_distance + pregrasp_distance,
            # )
            backup_distance = pushin_distance + pregrasp_distance
            # link_pose = env.obs.state.get_pose(key='block/|block/block')
            # grasp_pose = grasp_pose.transform(link_pose.matrix)
            rotmat = quaternions.quat2mat(grasp_pose.orientation)
            grasp_direction = rotmat @ np.array([0, 0, 1])
            grasp_backup_pose = Pose(
                position=grasp_pose.position + grasp_direction * backup_distance,
                orientation=grasp_pose.orientation,
            )
            try:
                # candidate.check_feasibility(env=env)
                self.check_feasibility(env=env, pose=grasp_pose)
                self.check_feasibility(env=env, pose=grasp_backup_pose)
                logging.info(
                    f"[{i}|{env.episode_id}] "
                    + f"pushin_distance: {pushin_distance} ({pushin_more})"
                )
                break
            except Action.InfeasibleAction as e:
                errors.append(e)
                grasp_pose = None
                # candidate = None
                continue
        del normals, link_point_cloud, grasp_to_ee, candidate_indices, p
        if grasp_pose is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"all candidates failed {self}:"
                + ",".join(list({str(e) for e in errors})[:3]),
            )

        

        ee_action_sequence = [
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=False,
                end_effector_position=grasp_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=True,  # now allow contact
                gripper_command=True,  # and close gripper
                end_effector_position=grasp_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
        ]
        if with_backup:
            ee_action_sequence.append(
                EndEffectorAction(
                    allow_contact=True,
                    gripper_command=True,
                    end_effector_position=grasp_backup_pose.position,
                    end_effector_orientation=get_best_orn_for_gripper(
                        reference_orn=euler.euler2quat(0, 0, 0),
                        query_orn=grasp_backup_pose.orientation,
                    ),
                    use_early_stop_checker=False,
                )
            )
        self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute_grasp(self, env, info={"log": "", "subtrajectories": []},):
        for i, ee_action in enumerate(self.ee_action_sequence):
            ee_action.execute_bebs(env, info=info)
        
        # chech if grasp objects
        state = env.get_obs().state
        right_finger_pos = state.robot_state.link_states['ur5e/|ur5e/base|ur5e/shoulder_link|ur5e/upper_arm_link|ur5e/forearm_link|ur5e/wrist_1_link|ur5e/wrist_2_link|ur5e/wrist_3_link|ur5e/wsg50/|ur5e/wsg50/base|ur5e/wsg50/right_finger'].pose.position
        left_finger_pos = state.robot_state.link_states['ur5e/|ur5e/base|ur5e/shoulder_link|ur5e/upper_arm_link|ur5e/forearm_link|ur5e/wrist_1_link|ur5e/wrist_2_link|ur5e/wrist_3_link|ur5e/wsg50/|ur5e/wsg50/base|ur5e/wsg50/left_finger'].pose.position
        distance_between_fingers = np.linalg.norm(left_finger_pos - right_finger_pos)
        is_success = (distance_between_fingers > griper_closed_threshold)
        return is_success

    def check_feasibility(self, env: Env, pose: Pose):
        # first check for kinematic reachability
        result = env.robot.inverse_kinematics(
            Pose(
                position=pose.position,
                orientation=get_best_orn_for_gripper(
                        reference_orn=euler.euler2quat(0, 0, 0),
                        query_orn=pose.orientation,
                    ),
            )
        )
        if result is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__, message="IK Failed"
            )

