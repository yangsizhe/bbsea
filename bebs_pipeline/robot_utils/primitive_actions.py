from scalingup.utils.core import (
    Pose,
    PointCloud,
    Env,
    Action,
    EndEffectorAction,
    DoNothingAction,
    normal_to_forward_quat,
    get_best_orn_for_gripper,
    EnvSamplerConfig,
    Task,
)
import numpy as np
import logging
from transforms3d import affines, euler, quaternions
from scipy.spatial import ConvexHull
import sys 
sys.path.append("..") 
from perception.object_detect import get_object_point_cloud


top_down_grasp_bias = 1e2
griper_closed_threshold = 0.02
prismatic_joint_distance = 0.25

# Push(obj_name), Pick(obj_name), PlaceOn(obj_name), PrismaticJointOpen(obj_name), PrismaticJointClose(obj_name), RevoluteJointOpen(obj_name), RevoluteJointClose(obj_name), Press(obj_name)
class Push():

    def __init__(self, obj_name, direction, distance) -> None:
        self.obj_name = obj_name
        direction = np.array(direction) 
        self.direction = direction / np.linalg.norm(direction)  # Normalize the direction
        self.distance = distance
        self.attempt_count = 0
        self.ee_action_sequence = None

    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('Push -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        # except Action.InfeasibleAction as e:
        #     logging.warning(e)
        #     return []
        except Exception as e:
            logging.warning(e)
            return []

    def point_cloud_to_actions(self, link_point_cloud: PointCloud, env: Env):
        object_center = np.mean(link_point_cloud.xyz_pts, axis=0)
        width = np.linalg.norm(np.max(link_point_cloud.xyz_pts, axis=0)[:2] - np.min(link_point_cloud.xyz_pts, axis=0)[:2])
        # Determine where the gripper should be positioned to start the push
        # We approach from the opposite direction
        approach_position = object_center - np.array([self.direction[0], self.direction[1], 0]) * (width + 0.02)
        
        # We will use the existing end effector's orientation for simplicity
        state = env.obs.state
        end_effector_orientation = state.end_effector_pose.orientation

        # Action sequence
        self.ee_action_sequence = [
            # Approach the object from the opposite side of the push direction
            EndEffectorAction(
                allow_contact=False,
                gripper_command=False,
                end_effector_position=approach_position,
                end_effector_orientation=end_effector_orientation,
                use_early_stop_checker=False,
            ),
            # Close the gripper to stabilize before pushing
            EndEffectorAction(
                allow_contact=True,
                gripper_command=True,
                end_effector_position=approach_position,
                end_effector_orientation=end_effector_orientation,
                use_early_stop_checker=False,
            ),
            # Push the object
            EndEffectorAction(
                allow_contact=True,
                gripper_command=True,  # Gripper remains closed
                end_effector_position=approach_position + np.array([self.direction[0], self.direction[1], 0]) * (self.distance + 0.05),  # We add the initial approach offset to ensure the object is pushed by the desired distance
                end_effector_orientation=end_effector_orientation,
                use_early_stop_checker=False,
            ),
        ]
        # self.attempt_count += 1

    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode

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

        action0 = Push('red block', direction=[-1, 0], distance=0.1)
        action0.do(env)


class Pick():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name
        self.attempt_count = 0
        self.ee_action_sequence = None
        self.is_success = False

    # def do(self, env, obs):
    #     link_point_cloud = get_object_point_cloud(env, self.obj_name)
    #     self.point_cloud_to_actions(link_point_cloud, env)
    #     sub_episode, obs = self.execute(env, obs)
    #     return sub_episode, obs
    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('Pick -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        # except Action.InfeasibleAction as e:
        #     logging.warning(e)
        #     return []
        except Exception as e:
            logging.warning(e)
            return []

    def point_cloud_to_actions(
        self,
        link_point_cloud: PointCloud, env: Env,
        with_backup=True, pushin_more=True
    ):
        state = env.obs.state
        numpy_random = env.policy_numpy_random

        # Determine the orientation using the cal_pick_orientation function
        # base_orientation = cal_pick_orientation(link_point_cloud)

        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts.mean(axis=0)
        grasp_to_ee /= np.linalg.norm(grasp_to_ee)
       
        pointing_down = link_point_cloud.normals[:, 2] < 0.0
        p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones pointing up more often
            p = np.exp(link_point_cloud.normals[:, 2] * top_down_grasp_bias)
        p /= p.sum()

        errors = []
        for i in range(env.config.num_action_candidate_attempt):
            candidate_indices = numpy_random.choice(
                len(link_point_cloud),
                size=min(env.config.num_action_candidates_for_average, len(link_point_cloud)),
                p=p,
                replace=True,
            )
            candidate_points = link_point_cloud.xyz_pts[candidate_indices].copy()
            position = candidate_points.mean(axis=0)
            position[2] = link_point_cloud.xyz_pts[:, 2].mean()
            grasp_to_ee = state.end_effector_pose.position - position
            grasp_to_ee /= np.linalg.norm(grasp_to_ee)

            # z_angle = numpy_random.uniform(np.pi, -np.pi)
            # z_orn = euler.euler2quat(0, 0, z_angle)
            # base_orientation = quaternions.qmult(base_orientation, z_orn)
            base_orientation = cal_pick_orientation(candidate_points)

            pregrasp_distance = numpy_random.uniform(0.1, 0.2)
            if pushin_more:
                pushin_distance = (len(candidate_indices) - self.attempt_count) / len(
                    candidate_indices
                ) * (
                    env.config.max_pushin_dist - env.config.min_pushin_dist
                ) + env.config.min_pushin_dist
            else:
                pushin_distance = (
                    self.attempt_count
                    / len(candidate_indices)
                    * (env.config.max_pushin_dist - env.config.min_pushin_dist)
                    + env.config.min_pushin_dist
                )


            normal = link_point_cloud.normals[candidate_indices[0]]
            if np.dot(grasp_to_ee, normal) < 0:
                normal *= -1.0
            grasp_pose = Pose(
                position=position - normal * pushin_distance,
                orientation=base_orientation,
            )

            backup_distance = pushin_distance + pregrasp_distance
            rotmat = quaternions.quat2mat(grasp_pose.orientation)
            grasp_direction = rotmat @ np.array([0, 0, 1])
            grasp_backup_pose = Pose(
                position=grasp_pose.position + grasp_direction * backup_distance,
                orientation=grasp_pose.orientation,
            )
            try:
                self.check_feasibility(env=env, pose=grasp_pose)
                self.check_feasibility(env=env, pose=grasp_backup_pose)
                break
            except Action.InfeasibleAction as e:
                errors.append(e)
                grasp_pose = None
                continue

        if grasp_pose is None:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"all candidates failed {self}:" + ",".join(list({str(e) for e in errors})[:3]),
            )
        pick_orientation = get_best_orn_for_gripper(
                                reference_orn=euler.euler2quat(0, 0, 0),
                                query_orn=grasp_pose.orientation,
                            )
        ee_action_sequence = [
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=False,
                end_effector_position=grasp_pose.position + normal * env.config.prepick_distance,
                end_effector_orientation=pick_orientation,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=False,
                end_effector_position=grasp_pose.position,
                end_effector_orientation=pick_orientation,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=True,  # now allow contact
                gripper_command=True,  # and close gripper
                end_effector_position=grasp_pose.position,
                end_effector_orientation=pick_orientation,
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
        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
   
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
       
        # chech if grasp objects
        state = env.obs.state
        right_finger_pos = state.robot_state.link_states['ur5e/|ur5e/base|ur5e/shoulder_link|ur5e/upper_arm_link|ur5e/forearm_link|ur5e/wrist_1_link|ur5e/wrist_2_link|ur5e/wrist_3_link|ur5e/wsg50/|ur5e/wsg50/base|ur5e/wsg50/right_finger'].pose.position
        left_finger_pos = state.robot_state.link_states['ur5e/|ur5e/base|ur5e/shoulder_link|ur5e/upper_arm_link|ur5e/forearm_link|ur5e/wrist_1_link|ur5e/wrist_2_link|ur5e/wrist_3_link|ur5e/wsg50/|ur5e/wsg50/base|ur5e/wsg50/left_finger'].pose.position
        distance_between_fingers = np.linalg.norm(left_finger_pos - right_finger_pos)
        self.is_success = (distance_between_fingers > griper_closed_threshold)

        return sub_episode

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


def cal_pick_orientation(candidate_points: np.array) -> np.array:
    # Project all points to the XY plane (top-down view)
    xy_points = candidate_points[:, :2]
   
    # Calculate the center of the points
    center = np.mean(xy_points, axis=0)

    # Calculate the convex hull of the points
    hull = ConvexHull(xy_points)

    # For each edge, compute the point on this edge that's closest to the centroid
    edges = [hull.points[simplex] for simplex in hull.simplices]
    closest_points_to_center = []
    for edge in edges:
        vector = edge[1] - edge[0]
        t = np.dot(center - edge[0], vector) / np.dot(vector, vector)
        t = np.clip(t, 0, 1)
        closest_point = edge[0] + t * vector
        closest_points_to_center.append(closest_point)
   
    # Select the edge for which the closest point is also closest to the centroid
    closest_point_index = np.argmin([np.linalg.norm(point - center) for point in closest_points_to_center])
    direction_vector = closest_points_to_center[closest_point_index] - center
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # The direction_vector provides the direction in which the gripper should be oriented for the grasp
    angle = np.arctan2(direction_vector[1], direction_vector[0])

    # Rotate the angle by 90 degrees (π/2 radians)
    angle += np.pi / 2

    # Convert the angle to a quaternion representing the rotation around the Z-axis
    orientation = euler.euler2quat(0, 0, angle)

    return orientation


class PlaceOn():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name
        self.attempt_count = 0
        self.ee_action_sequence = None
    
    # def do(self, env, obs):
    #     link_point_cloud = get_object_point_cloud(env, self.obj_name)
    #     self.point_cloud_to_actions(link_point_cloud, env)
    #     sub_episode, obs = self.execute(env, obs)
    #     return sub_episode, obs
    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('PlaceOn -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        except Action.InfeasibleAction as e:
            logging.warning(e)
            return []
        

    def point_cloud_to_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
    ):
        state = env.obs.state
        numpy_random = env.policy_numpy_random

        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        filter_out_mask = normals[:, 2] < env.config.pointing_up_normal_threshold

        xyz_pts = link_point_cloud.xyz_pts[~filter_out_mask]
        normals = normals[~filter_out_mask]
        # candidate: Action = DoNothingAction()
        # link_pose = obs.state.get_pose(key=self.link_path)
        # TODO sampling can be biased towards points that belong to larger surface areas
        num_candidates = (~filter_out_mask).sum()
        if num_candidates == 0:
            raise Action.InfeasibleAction(
                action_class_name=type(self).__name__,
                message=f"no candidates for {self.obj_name}",
            )
        logging.debug(f"PlaceOnLinkAction has {num_candidates} candidates")
        # for attempt, idx in enumerate(
        #     numpy_random.choice(
        #         num_candidates, size=env.config.num_action_candidates, replace=True
        #     )
        # ):
        for i in range(env.config.num_action_candidate_attempt):
            candidate_indices = numpy_random.choice(
                num_candidates,
                size=min(env.config.num_action_candidates_for_average, num_candidates),
                replace=True,
            )
            candidate_points = xyz_pts[candidate_indices].copy()
            position = candidate_points.mean(axis=0)
            # compute base orientation, randomize along Z-axis
            normal = normals[candidate_indices].copy().mean(axis=0)
            normal /= np.linalg.norm(normal)
            # import pdb;pdb.set_trace()
            # normal = np.array([0.0, 0.0, 1.0])
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
                + normal * 0.15,
                # * numpy_random.uniform(
                #     env.config.place_height_min, env.config.place_height_max
                # ),
                orientation=base_orientation,
            )
            # NOTE: if too low, then gripper + object can also bump and move the
            # object it's placing on top of, like the previously stacked blocks or
            # drawer
            preplace_dist = numpy_random.uniform(
                env.config.preplace_dist_min, env.config.preplace_dist_max
            )
            rotmat = quaternions.quat2mat(place_pose.orientation)
            place_direction = rotmat @ np.array([0, 0, 1])
            # preplace_pos = place_pose.position + place_direction * preplace_dist 
            preplace_pos = place_pose.position
            preplace_pos[2] = np.max(link_point_cloud.xyz_pts[:, 2])
            preplace_pos = preplace_pos + place_direction * preplace_dist + np.array([0.0, 0.0, 0.1])  # find the highest point
            postplace_pos = preplace_pos + np.array([0.0, 0.0, 0.1])
            orn = get_best_orn_for_gripper(
                reference_orn=euler.euler2quat(0, 0, 0),
                query_orn=place_pose.orientation,
            )
            try:
                self.check_feasibility(env=env, pose=place_pose, orn=orn, preplace_pos=preplace_pos)
                logging.info(f"[{i}] preplace_dist: {preplace_dist}")
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
                end_effector_position=postplace_pos,
                end_effector_orientation=orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
        ]
        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode
        
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
    
class PlaceAt():

    def __init__(self, place_pos: list):
        self.place_pos = np.array(place_pos)
        self.ee_action_sequence = None
    
    def do(self, env: Env):
        try:
            self.place_pos_to_actions(self.place_pos, env)
            sub_episode = self.execute(env)
            return sub_episode
        except:
            return []

    def place_pos_to_actions(
        self, 
        place_pos: np.ndarray, env: Env, 
    ):
        numpy_random = env.policy_numpy_random

        place_orn = euler.euler2quat(0, 0, 0)
        rotmat = quaternions.quat2mat(place_orn)
        place_direction = rotmat @ np.array([0, 0, 1])
        preplace_dist = numpy_random.uniform(
            env.config.preplace_dist_min, env.config.preplace_dist_max
        )
        preplace_pos = place_pos + place_direction * preplace_dist
        ee_action_sequence =  [
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=preplace_pos,
                end_effector_orientation=place_orn,
                allow_contact=False,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=True,
                end_effector_position=place_pos,
                end_effector_orientation=place_orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=place_pos,
                end_effector_orientation=place_orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                gripper_command=False,
                end_effector_position=preplace_pos,
                end_effector_orientation=place_orn,
                allow_contact=True,
                use_early_stop_checker=False,
            ),
        ]
        self.ee_action_sequence = ee_action_sequence
    
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode
        
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


class PrismaticJointOpen():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name
        self.attempt_count = 0
        self.ee_action_sequence = None
    
    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('PrismaticJointOpen -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        except Exception as e:
            logging.warning(e)
            return []
        
    def point_cloud_to_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
        pushin_more=True
    ):
        state = env.obs.state
        numpy_random = env.policy_numpy_random
        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        pointing_down = normals[:, 2] < 0.0
        p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones point up less often
            # p = np.exp(normals[:, 2] * top_down_grasp_bias)
            p = np.exp(top_down_grasp_bias * np.sum(np.square(normals[:, :2]), axis=1))
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
            backup_distance = prismatic_joint_distance # pushin_distance + pregrasp_distance
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
            EndEffectorAction(
                allow_contact=True,
                gripper_command=True,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=True,
                gripper_command=False,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            )
        ]

        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode

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


class PrismaticJointClose():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name
        self.attempt_count = 0
        self.ee_action_sequence = None
    
    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('PrismaticJointClose -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        except Exception as e:
            logging.warning(e)
            return []
        
    def point_cloud_to_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
        pushin_more=True
    ):
        state = env.obs.state
        numpy_random = env.policy_numpy_random
        grasp_to_ee = state.end_effector_pose.position - link_point_cloud.xyz_pts
        grasp_to_ee /= np.linalg.norm(grasp_to_ee, axis=1)[:, None]
        normals = link_point_cloud.normals
        pointing_down = normals[:, 2] < 0.0
        p = np.ones(shape=len(link_point_cloud), dtype=np.float64)
        if not pointing_down.all():
            # sample the ones point up less often
            # p = np.exp(normals[:, 2] * top_down_grasp_bias)
            p = np.exp(top_down_grasp_bias * np.sum(np.square(normals[:, :2]), axis=1))
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
            backup_distance = prismatic_joint_distance # pushin_distance + pregrasp_distance
            # link_pose = env.obs.state.get_pose(key='block/|block/block')
            # grasp_pose = grasp_pose.transform(link_pose.matrix)
            rotmat = quaternions.quat2mat(grasp_pose.orientation)
            grasp_direction = rotmat @ np.array([0, 0, 1])

            grasp_backup_pose = Pose(
                position=grasp_pose.position - grasp_direction * backup_distance,
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
            EndEffectorAction(
                allow_contact=True,
                gripper_command=True,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=True,
                gripper_command=False,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            )
        ]

        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode

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

class RevoluteJointOpen():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name

class RevoluteJointClose():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name

class Press():

    def __init__(self, obj_name) -> None:
        self.obj_name = obj_name
        self.attempt_count = 0
        self.ee_action_sequence = None
    
    def do(self, env: Env):
        try:
            link_point_cloud = get_object_point_cloud(env, self.obj_name)
        except:
            print('Press -- no object')
            return []
        try:
            self.point_cloud_to_actions(link_point_cloud, env)
            sub_episode = self.execute(env)
            return sub_episode
        except Exception as e:
            logging.warning(e)
            return []
        
    def point_cloud_to_actions(
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
            grasp_pose = Pose(
                position=position - normal * pushin_distance,
                orientation=base_orientation,
            )
            backup_distance = pushin_distance + pregrasp_distance
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
                allow_contact=True,
                gripper_command=True,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=True,
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
        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute(self, env, info={"log": "", "subtrajectories": []},):
        sub_episode = []
        for i, ee_action in enumerate(self.ee_action_sequence):
            trajectory_step = ee_action.execute_bebs(env, info=info)
            sub_episode.append(trajectory_step)
        
        return sub_episode

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

         
class PrismaticJointAction():

    def __init__(self) -> None:
        self.attempt_count = 0
        self.ee_action_sequence = None

    def point_cloud_to_prismatic_joint_actions(
        self, 
        link_point_cloud: PointCloud, env: Env, 
        is_open, pushin_more=True
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
            if is_open:
                grasp_backup_pose = Pose(
                    position=grasp_pose.position + grasp_direction * backup_distance,
                    orientation=grasp_pose.orientation,
                )
            else:
                grasp_backup_pose = Pose(
                    position=grasp_pose.position - grasp_direction * backup_distance,
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
            EndEffectorAction(
                allow_contact=True,
                gripper_command=True,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=True,
                gripper_command=False,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            )
        ]

        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute_prismatic_joint_action(self, env, info={"log": "", "subtrajectories": []},):
        for i, ee_action in enumerate(self.ee_action_sequence):
            ee_action.execute_bebs(env, info=info)

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
        

class PokeAction():

    def __init__(self) -> None:
        self.attempt_count = 0
        self.ee_action_sequence = None

    def point_cloud_to_poke_actions(
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
            grasp_pose = Pose(
                position=position - normal * pushin_distance,
                orientation=base_orientation,
            )
            backup_distance = pushin_distance + pregrasp_distance
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
                allow_contact=True,
                gripper_command=True,
                end_effector_position=grasp_backup_pose.position,
                end_effector_orientation=get_best_orn_for_gripper(
                    reference_orn=euler.euler2quat(0, 0, 0),
                    query_orn=grasp_backup_pose.orientation,
                ),
                use_early_stop_checker=False,
            ),
            EndEffectorAction(
                allow_contact=False,  # Don't allow contact before activating gripper
                gripper_command=True,
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
        # self.attempt_count += 1
        self.ee_action_sequence = ee_action_sequence
    
    def execute_poke(self, env, info={"log": "", "subtrajectories": []},):
        for i, ee_action in enumerate(self.ee_action_sequence):
            ee_action.execute_bebs(env, info=info)

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
