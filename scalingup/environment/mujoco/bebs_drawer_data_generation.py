import time
import numpy as np

from scalingup.environment.mujoco.mujocoEnv import MujocoUR5EnvFromObjConfigList
from scalingup.environment.mujoco.table_top import TableTopMujocoEnv
from scalingup.environment.mujoco.utils import MujocoObjectInstanceConfig
from scalingup.utils.core import DegreeOfFreedomRange


class TableTopPickAndPlace(TableTopMujocoEnv, MujocoUR5EnvFromObjConfigList):
    pass


class DrawerMujocoEnv(TableTopMujocoEnv):
    def __init__(
        self,
        pose_randomization: bool = False,
        position_randomization: bool = False,
        table_length: float = 1.165,
        table_width: float = 0.58,
        table_x_pos: float = 0.4,
        drawer_xml_path: str = "scalingup/environment/mujoco/assets/bebs_custom/small_drawer_open.xml",
        **kwargs,
    ):
        setup_numpy_random = np.random.RandomState(seed=int(time.time()))
        self.drawer_xml_path = drawer_xml_path
        self.kwargs = kwargs
        x_pos = setup_numpy_random.uniform(0.0, 0.2)
        y_pos = setup_numpy_random.uniform(0.0, 0.7)
        z_rot = setup_numpy_random.uniform(low=-np.pi / 5, high=0)
        configs = [
            MujocoObjectInstanceConfig(
                obj_class="drawer",
                asset_path=self.drawer_xml_path,
                qpos_range=[
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    DegreeOfFreedomRange(upper=0.1, lower=0.0), 
                ],
                position=(x_pos, y_pos, 0.05),
                euler=(0, 0, z_rot),
            ),
        ]

        super().__init__(
            obj_instance_configs=configs,
            **self.kwargs,
        )
    
    def reset(self, episode_id: int = 0):
        setup_numpy_random = np.random.RandomState(seed=episode_id)
        x_pos = setup_numpy_random.uniform(0.0, 0.1)
        y_pos = setup_numpy_random.uniform(0.4, 0.7)
        z_rot = setup_numpy_random.uniform(low=-np.pi / 5, high=0)
        configs = [
            MujocoObjectInstanceConfig(
                obj_class="drawer",
                asset_path=self.drawer_xml_path,
                qpos_range=[
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),
                    DegreeOfFreedomRange(upper=0.0, lower=0.0),

                    # DegreeOfFreedomRange(upper=-0.176, lower=-0.176),  
                    DegreeOfFreedomRange(upper=0, lower=-0.08), 
                ],
                position=(x_pos, y_pos, 0.05),
                euler=(0, 0, z_rot),
            ),
        ]

        super().__init__(
            obj_instance_configs=configs,
            **self.kwargs,
        )
        return super().reset(episode_id=episode_id)
