import os
import gc
import logging
import sys
import typing
import yaml
import hydra
import torch
import numpy as np

from scalingup.algo.diffusion import DiffusionScalingUpAlgo, DiffusionPolicy
from scalingup.data.dataset import StateTensor
from scalingup.data.window_dataset import (
    ControlActionSequenceTensor,
    ControlActionTensor,
    StateSequenceTensor,
    TrajectoryWindowDataset,
    TrajectoryWindowTensor,
)
from scalingup.policy.scalingup import ScalingUpDataGen
from scalingup.utils.generic import setup_logger
from scalingup.utils.core import (
    Env,
    EnvSampler,
    EnvSamplerConfig,
    TaskSampler,
    Trajectory,
    PointCloud,
)
import psutil
import requests
from retry import retry
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_base = os.getenv("OPENAI_API_ENDPOINT")
# openai.api_type = os.getenv("OPENAI_API_TYPE")  #'azure'
# openai.api_version = os.getenv("OPENAI_API_VERSION")  # '2023-05-15' # this may change in the future


def if_early_end(success_num, i, config):
    success_rate = 1.0
    if (i + 1) % 50 == 0:
        success_rate = success_num / i
    return success_num >= config.success_trajectory_number_per_task or success_rate < 0.01

def make_output_dirs_and_filter_task_desc_list(config, task_desc_list):
    for task_desc in task_desc_list:
        for path in [f'{config.log_path}/{config.scene_id}/{task_desc}', f'{config.trajectory_path}/{config.scene_id}/{task_desc}']:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except:
                    try:
                        task_desc_list.remove(task_desc)
                    except:
                        print('the task has been removed from the list')
    return task_desc_list

def make_trajectory_dir_for_each_episode(trajectory_dir_path):
    is_the_trajectory_sucess = False
    if not os.path.exists(trajectory_dir_path):
        os.makedirs(trajectory_dir_path)
        return False, is_the_trajectory_sucess
    else:  # have collected
        file_list = os.listdir(trajectory_dir_path)

        if len(file_list) == 0:
            return False, is_the_trajectory_sucess
        elif 'True' in file_list[0]:
            is_the_trajectory_sucess = True
            return True, is_the_trajectory_sucess
        else:
            is_the_trajectory_sucess = False
            return True, is_the_trajectory_sucess

def save_str_to_file(content, file_name):
    f = open(file_name, 'w')
    f.write(content)
    f.close()

@retry(tries=5, delay=1, backoff=2, max_delay=120)
def call_openai_api(prompt, temperature=0.2):
    messages=[{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine='gpt-4',
        messages=messages,
        temperature=temperature,
    )
    return response

def get_link_point_cloud(env, link_path):
    MAX_OFFSCREEN_HEIGHT = 640
    res = (
        (
            np.array(env.config.obs_dim)
            / max(env.config.obs_dim)
            * MAX_OFFSCREEN_HEIGHT
        )
        .astype(int)
        .tolist()
    )
    images = env.render(obs_dim=(res[0], res[1]))
    # mask = images['front'].point_cloud.segmentation_pts['block/|block/block']
    # mask = mask.reshape(426, 640) * 255
    # Image.fromarray(mask.astype(np.uint8)).save('mask.jpg')
    # # cv2.imwrite('mask.jpg', mask.reshape(426, 640))
    # for sensor_output in images.values():
    #     sensor_output.point_cloud[link_path]

    # combine all point clouds and get out the relevant link
    link_point_clouds = [
        sensor_output.point_cloud[link_path] for sensor_output in images.values()
    ]
    link_point_cloud: PointCloud = sum(
        link_point_clouds[1:], start=link_point_clouds[0]
    )
    return link_point_cloud

def memory_usage() -> int:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size


def set_up_env(
    config_name: str, extra_overrides: typing.Optional[typing.List[str]] = None
) -> typing.Tuple[Env, TaskSampler, EnvSamplerConfig]:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # camera config
    with hydra.initialize(
        config_path="../scalingup/config",
        job_name="task_inference",
        version_base="1.2",
    ):
        cfg = hydra.compose(config_name="common")
        obs_dim = cfg.obs_dim
        obs_cameras = cfg.obs_cameras
        visibility_checker_cam_name = cfg.main_cam
    # env config
    with hydra.initialize(
        config_path="../scalingup/config/evaluation",
        job_name="task_inference",
        version_base="1.2",
    ):
        overrides = [
            f"env.config.obs_cameras={obs_cameras}",
            f"env.config.obs_dim={obs_dim}",
        ]
        if extra_overrides is not None:
            overrides.extend(extra_overrides)
        cfg = hydra.compose(
            config_name=config_name,
            overrides=overrides,
        )
        # instantiate env, task_sampler, sampler_config
        env: Env = hydra.utils.instantiate(
            cfg.env,
            discretizer=None,
            visibility_checker_cam_name=visibility_checker_cam_name,
        )
        task_sampler: TaskSampler = hydra.utils.instantiate(cfg.task_sampler)
        sampler_config: EnvSamplerConfig = hydra.utils.instantiate(cfg.sampler_config)
        return env, task_sampler, sampler_config
    


OBJECT_NAMES = []
LINK_NAMES_AND_LINK_PATHS = {}
OBJECTS_NUM = {}

def set_up_scene(scene_id):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(config_path="./config", version_base="1.2"):
        cfg = hydra.compose(config_name='environment', overrides=[f'scene="{scene_id}"'])
    # instantiate env, sampler_config
    env: Env = hydra.utils.instantiate(
        cfg.env,
        discretizer=None,
    )
    sampler_config: EnvSamplerConfig = hydra.utils.instantiate(cfg.sampler_config)
    env.env_sampler_config = sampler_config

    # load the name of all the objects
    OBJECT_NAMES.clear()
    if cfg['env']['assets'] is not None:
        for item in cfg['env']['assets']:
            if item['number'] > 1:
                for id in range(item['number']):
                    OBJECT_NAMES.append(f"{item['name']} {id}")
            elif item['number'] == 1:
                OBJECT_NAMES.append(f"{item['name']}")
    if cfg['env']['fixed_objects'] is not None:
        for item in cfg['env']['fixed_objects']:
            # OBJECT_NAMES.append(f"{item['name']}")
            LINK_NAMES_AND_LINK_PATHS[item['name']] = cfg.links_of_fixed_object_to_be_detected[item['name']].path
            for link in cfg.links_of_fixed_object_to_be_detected[item['name']].links:
                LINK_NAMES_AND_LINK_PATHS[link['name']] = link['link_path']
    return env


def set_up_policy(
    policy_config_name: str = "scalingup", logging_level: int = logging.INFO
):
    setup_logger(logging_level=logging_level)
    with hydra.initialize(
        config_path="../scalingup/config/policy",
        job_name="task_inference",
        version_base="1.2",
    ):
        cfg = hydra.compose(config_name=policy_config_name)
        policy: ScalingUpDataGen = hydra.utils.instantiate(cfg)
    return policy


def get_trajectory_window_dataset(path: str):
    with hydra.initialize(
        config_path="../scalingup/config",
        job_name="task_inference",
        version_base="1.2",
    ):
        cfg = hydra.compose(
            config_name="train_offline",
            overrides=[f"dataset_path={path}", "evaluation=table_top_bus_balance"],
        )
        return typing.cast(
            TrajectoryWindowDataset, hydra.utils.instantiate(cfg.algo.replay_buffer)
        )


def default_sampler_config(
    max_time: typing.Optional[float] = None,
    obs_horizon: typing.Optional[int] = None,
    overrides: typing.Optional[typing.List[str]] = None,
):
    with hydra.initialize(
        config_path="../scalingup/config/evaluation/sampler_config",
        job_name="task_inference",
        version_base="1.2",
    ):
        if overrides is None:
            overrides = []
        if max_time is not None:
            overrides.append(f"max_time={max_time}")
        if obs_horizon is not None:
            overrides.append(f"obs_horizon={obs_horizon}")
        cfg = hydra.compose(
            config_name="default",
            overrides=overrides,
        )
        return typing.cast(EnvSamplerConfig, hydra.utils.instantiate(cfg))


def get_sample_trajectory(
    config_name: str = "open_top_drawer",
    episode_id: int = 0,
    policy_config_name: str = "scalingup",
    assert_successful: bool = False,
    **kwargs,
) -> Trajectory:
    setup_logger()
    env, task_sampler = set_up_env(config_name=config_name)
    policy = set_up_policy(policy_config_name=policy_config_name)
    sampler = EnvSampler(env=env, task_sampler=task_sampler)
    env_sample = sampler.sample(
        policy=policy,
        episode_id=episode_id,
        return_trajectory=True,
        config=default_sampler_config(**kwargs),
    )
    traj = env_sample.trajectory
    assert traj is not None
    traj.dump_video("successful_traj.mp4", add_text=False)
    if assert_successful:
        assert traj.task.check_success(traj=traj)
    return traj


def get_test_diffusion_algo(
    optimization_steps: int,
    dataset_path: str = "scalingup/wandb/latest-run/files",
    epochs: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    precache_text_descs: typing.Optional[typing.List[str]] = None,
):
    with hydra.initialize(config_path="../scalingup/config", version_base="1.2"):
        conf = hydra.compose(
            config_name="train_offline",
            overrides=[
                "algo=diffusion_default",
                f"dataset_path={dataset_path}",
                f"trainer.max_epochs={epochs}",
                f"algo.replay_buffer.num_steps_per_update={optimization_steps}",
                f"algo.replay_buffer.batch_size={batch_size}",
                f"algo.replay_buffer.num_workers={num_workers}",
                "evaluation=table_top_sorting",
                "algo.state_sequence_encoder.precache_text_descs="
                + (str(precache_text_descs) if precache_text_descs else "null"),
            ],
        )
        algo: DiffusionScalingUpAlgo = hydra.utils.instantiate(
            conf.algo,
        )  # fake action normalization
    logging.info("creating fake action normalization")
    algo.window_replay_buffer.max_action_values = torch.ones((algo.action_dim,)).float()
    algo.window_replay_buffer.min_action_values = -torch.ones((algo.action_dim,)).float()
    algo.window_replay_buffer.control_frequencies = [5]
    return algo


def get_test_diffusion_policy(**kwargs) -> DiffusionPolicy:
    algo = get_test_diffusion_algo(optimization_steps=1)
    return typing.cast(DiffusionPolicy, algo.get_policy(**kwargs))


def get_test_trajectory_window(
    window_size: int,
    batch_size: int,
    action_dim: int,
    state_tensor: StateTensor,
    observation_horizon: int,
    task_name: str,
) -> TrajectoryWindowTensor:
    attrs = list(StateTensor.__annotations__.keys())
    return TrajectoryWindowTensor(
        state_sequence=StateSequenceTensor(
            sequence=[
                StateTensor(
                    **torch.utils.data.default_collate(  # type: ignore
                        [
                            {attr: getattr(state_tensor, attr) for attr in attrs}
                            for i in range(batch_size)
                        ]
                    )
                )
            ]
            * observation_horizon
        ),
        action_sequence=ControlActionSequenceTensor(
            sequence=[
                ControlActionTensor(
                    value=torch.randn((batch_size, action_dim)),
                )
                for _ in range(window_size)
            ]
        ),
        task_metrics=torch.randint(0, 1, size=(batch_size, 1)),
        task_names=[task_name for _ in range(batch_size)],
    )
