from environment.table_top import TableTopMujocoEnv
from environment.utils import MujocoObjectInstanceConfig
from scalingup.utils.core import (
    QPosRange,
    DegreeOfFreedomRange,
)
from dm_control import mjcf
from dm_control.mjcf import RootElement
import numpy as np
import logging
import typing
import numpy as np
from scalingup.environment.mujoco.mujocoEnv import (
    MujocoEnv,
    MujocoUR5EnvFromObjConfigList,
)
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.environment.mujoco.utils import (
    MujocoObjectColorConfig,
    MujocoObjectInstanceConfig,
)
from scalingup.utils.core import (
    ControlAction,
    DegreeOfFreedomRange,
    Env,
    JointState,
    JointType,
)
from scalingup.utils.state_api import check_joint_activated

class TableTopEnv(TableTopMujocoEnv):
    def __init__(self, assets, fixed_objects, **kwargs):
        self.end_after_activated_time = 4
        self.assets = assets
        self.fixed_objects_names = []
        if fixed_objects is None:
            super().__init__(
                obj_instance_configs=[],
                dynamic_mujoco_model=True,
                **kwargs,
            )
            return
        fixed_objects_names = [fixed_object['name'] for fixed_object in fixed_objects]
        self.fixed_objects_names = fixed_objects_names
        configs = []
        if 'drawer' in fixed_objects_names:
            configs.append(
                MujocoObjectInstanceConfig(
                    obj_class="drawer",
                    asset_path="environment/assets/bebs_custom/small_drawer.xml",
                    qpos_range=[
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    ],
                    position=(0.0, 0.7, 0.05),
                    euler=(0, 0, -np.pi / 5),
                )
            )
        if 'catapult' in fixed_objects_names:
            configs.append(
                MujocoObjectInstanceConfig(
                    obj_class="catapult",
                    asset_path="environment/assets/bebs_custom/catapult.xml",
                    qpos_range=[
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    ],
                    position=(0.55, 0.25, 0.075),
                    # euler=(0, 0, -np.pi / 5),
                )
            )
        if 'cupboard' in fixed_objects_names:
            configs.append(
                MujocoObjectInstanceConfig(
                    obj_class="cupboard",
                    asset_path="environment/assets/bebs_custom/cupboard.xml",
                    qpos_range=[
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    ],
                    position=(0.0, -0.8, 0.05),
                    euler=(0, 0, np.pi / 3),
                )
            )
        if 'mailbox' in fixed_objects_names:
            configs.append(
                MujocoObjectInstanceConfig(
                    obj_class="mailbox",
                    asset_path="environment/assets/bebs_custom/wall_mailbox/wall_mailbox.xml",
                    qpos_range=[
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                        DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    ],
                    position=(0.1, -0.4, 0.05),
                    euler=(0, 0, np.pi * 2 / 3),
                )
            )
        super().__init__(
            obj_instance_configs=configs,
            dynamic_mujoco_model=True,
            **kwargs,
        )

        if 'catapult' in fixed_objects_names:
            def catapult_mechanism(env: Env):
                """
                on top of checking if the catapult is activated, this function also
                makes sure the block is not picked and placed into the box
                """
                assert issubclass(type(env), TableTopEnv)
                catapult_env = typing.cast(TableTopEnv, env)
                physics = catapult_env.mj_physics
                # check pick and place first
                # grasp_id = catapult_env.mujoco_robot.get_grasped_obj_id(physics)
                # ee_pose = catapult_env.mujoco_robot.end_effector_pose
                # if grasp_id != -1 and ee_pose.position[1] < 0:
                #     logging.info("policy attempting to place block into box")
                #     env.done = True
                #     return

                catapult_joint = physics.model.joint("catapult/button_slider")
                min_value, max_value = catapult_joint.range
                activated = check_joint_activated(
                    joint_state=JointState(
                        current_value=physics.data.qpos[catapult_joint.qposadr[0]],
                        min_value=min_value,
                        max_value=max_value,
                        # VALUES BELOW NOT USED
                        name="",
                        joint_type=JointType.PRISMATIC,
                        axis=(0, 0, 0),
                        position=(0, 0, 0),
                        orientation=np.array([0, 0, 0, 0]),
                        parent_link="",
                        child_link="",
                    )
                )
                if (
                    activated
                    and self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active
                ):
                    logging.info("catapult activated")
                    self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active = 0
                    ctrl_cycles = int(
                        np.ceil(self.end_after_activated_time * env.config.ctrl.frequency)
                    )
                    ctrl_val = self.control_buffer.get_target_ctrl(t=self.time)[1].copy()
                    executed_ctrl_mask = (
                        self.control_buffer.timestamps
                        <= self.time + 1 / env.config.ctrl.frequency
                    )

                    # Cancel all pending controls
                    self.control_buffer = ControlAction(
                        value=self.control_buffer.value[executed_ctrl_mask],
                        timestamps=self.control_buffer.timestamps[executed_ctrl_mask],
                        config=self.config.ctrl,
                        target_ee_actions=self.control_buffer.target_ee_actions[
                            : sum(executed_ctrl_mask)
                        ]
                        if self.control_buffer.target_ee_actions is not None
                        else None,
                    )

                    start_time = self.control_buffer.timestamps[-1]
                    ctrl = ControlAction(
                        value=np.stack([ctrl_val] * ctrl_cycles),
                        timestamps=np.linspace(
                            start_time,
                            start_time + ctrl_cycles / env.config.ctrl.frequency,
                            ctrl_cycles,
                            endpoint=False,
                        )
                        + 1 / env.config.ctrl.frequency,
                        config=env.config.ctrl,
                        target_ee_actions=[
                            self.control_buffer.get_target_ctrl(t=self.time)[2]
                        ]
                        * ctrl_cycles,
                    )
                    env.control_buffer = env.control_buffer.combine(ctrl)
                    env.done = True

            self.step_fn_callbacks["catapult_mechanism"] = (
                int(1 / self.dt),
                catapult_mechanism,
            )

    def randomize(self):
        if 'catapult' in self.fixed_objects_names:
            self.mj_physics.model.ptr.eq("catapult/catapult_trigger").active = 1
        return super().randomize()

    def setup_objs(self, world_model: RootElement) -> QPosRange:
        obj_qpos_ranges = super().setup_objs(world_model=world_model)
        for i, asset in enumerate(self.assets):
            if asset['number'] > 1:
                for id in range(asset['number']):
                    asset_model = self.rename_model(mjcf.from_path(asset['model_path']), name=f"{asset['name']} {id}")
                    obj_body = self.add_obj_from_model(
                                    obj_model=asset_model,
                                    world_model=world_model,
                                    add_free_joint=True,
                                )
                    for geom_element in obj_body.find_all("geom"):
                        geom_element.condim = "4"
                        geom_element.friction = "0.9 0.8"
                    obj_qpos_ranges.extend(
                            [
                                DegreeOfFreedomRange(
                                    lower=qpos_range[0],
                                    upper=qpos_range[1]
                                    )
                                for qpos_range in asset['qpos_ranges']
                            ]
                        )
            elif asset['number'] == 1:
                asset_model = self.rename_model(mjcf.from_path(asset['model_path']), name=f"{asset['name']}")
                obj_body = self.add_obj_from_model(
                                obj_model=asset_model,
                                world_model=world_model,
                                add_free_joint=True,
                            )
                for geom_element in obj_body.find_all("geom"):
                    geom_element.condim = "4"
                    geom_element.friction = "0.9 0.8"
                obj_qpos_ranges.extend(
                        [
                            DegreeOfFreedomRange(
                                lower=qpos_range[0],
                                upper=qpos_range[1]
                                )
                            for qpos_range in asset['qpos_ranges']
                        ]
                    )

        return obj_qpos_ranges




























        # configs = []
        # for i, asset in enumerate(assets):
        #     configs.append(
        #         MujocoObjectInstanceConfig(
        #             obj_class=asset['name'],
        #             asset_path=asset['model_path'],
        #             qpos_range=[
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['x'][0],
        #                     lower=asset['pose_range']['x'][1]
        #                     ),
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['y'][0],
        #                     lower=asset['pose_range']['y'][1]
        #                     ),
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['z'][0],
        #                     lower=asset['pose_range']['z'][1]
        #                     ),
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['roll'][0],
        #                     lower=asset['pose_range']['roll'][1]
        #                     ),
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['pitch'][0],
        #                     lower=asset['pose_range']['pitch'][1]
        #                     ),
        #                 DegreeOfFreedomRange(
        #                     upper=asset['pose_range']['yaw'][0],
        #                     lower=asset['pose_range']['yaw'][1]
        #                     ),
        #             ],
        #             add_free_joint=True,
        #         ),
        #     )

        
                    # [
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][0][0],
                    #         lower=asset['qpos_ranges'][0][1]
                    #         ),
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][1][0],
                    #         lower=asset['qpos_ranges'][1][1]
                    #         ),
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][2][0],
                    #         lower=asset['qpos_ranges'][2][1]
                    #         ),
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][3][0],
                    #         lower=asset['qpos_ranges'][3][1]
                    #         ),
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][4][0],
                    #         lower=asset['qpos_ranges'][4][1]
                    #         ),
                    #     DegreeOfFreedomRange(
                    #         upper=asset['qpos_ranges'][5][0],
                    #         lower=asset['qpos_ranges'][5][1]
                    #         ),
                    # ]

