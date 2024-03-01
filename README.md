# BBSEA: An Exploration of Brain-Body Synchronization for Embodied Agents

Implementation of **BBSEA** from [BBSEA: An Exploration of Brain-Body Synchronization for Embodied Agents](https://arxiv.org/abs/2402.08212).

## Instructions

### Install dependencies

```
conda create -n bbsea python=3.8
conda activate bbsea
pip install torch
git clone git@github.com:yangsizhe/CLIP.git
cd CLIP
pip install -e .
pip install -e .
```

### Run the pipeline to generate demonstrations

```
export OPENAI_API_KEY=your_OPENAI_API_KEY
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python bebs_pipeline/bebs_pipeline.py max_trajectory_number_per_task=2000 success_trajectory_number_per_task=200 output_path=your_path_to_output scene_id=1
```

### Train a multi-task policy

```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python scalingup/train.py algo=diffusion_default evaluation.num_episodes=40 algo.replay_buffer.batch_size=256 trainer.max_epochs=1 evaluation=drawer dataset_path=your_path_to_dataset
```
`your_path_to_dataset` can be the `your_path_to_output` when run the pipeline to generate demonstrations.

### Inference

```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python scalingup/inference.py evaluation.num_episodes=10 policy=scalingup evaluation=drawer evaluation.start_episode=100000 policy.path=/path/to/your/checkpoint.ckpt
```

## License & Acknowledgements
BBSEA is licensed under the MIT license. MuJoCo is licensed under the Apache 2.0 license. 

We utilize the official implementation of [scalingup](https://github.com/real-stanford/scalingup) as codebase.

## Citation
If you find our work useful, please consider citing:
```
@article{yang2024bbsea,
  title={BBSEA: An Exploration of Brain-Body Synchronization for Embodied Agents},
  author={Yang, Sizhe and Luo, Qian and Pani, Anumpam and Yang, Yanchao},
  journal={arXiv preprint arXiv:2402.08212},
  year={2024}
}
```
