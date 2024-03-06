# Constrained Policy Optimization with Explicit Behavior Density

Official implementation for NeurIPS 2023 paper [Constrained Policy Optimization with Explicit Behavior Density for Offline Reinforcement Learning](https://arxiv.org/abs/2301.12130).


## Implementation requirement

1. Install [MuJoCo version 2.1.0](https://github.com/google-deepmind/mujoco/releases?page=2)
2. Install [D4RL](https://github.com/Farama-Foundation/D4RL/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688)
3. Install [rlkit][https://github.com/vitchyr/rlkit] 


### Getting Started

Then in order to run CPED, an example command is:

```
python examples/flow_hdf5_d4rl.py --env='halfcheetah-medium-v2' --policy_lr=1e-4 --model_type='nice --flowgan_train_epoch=500'
python examples/flow_hdf5_d4rl.py --env='antmaze-umaze-v2' --policy_lr=3e-4 --model_type='nice --flowgan_train_epoch=300'
```

For the antmaze-large tasks, the traing epoch of flowgan should be increased to 400 or 500 by specifying the hyperparameter flowgan_train_epoch=300 or flowgan_train_epoch=300.


## Citation

If you find this code useful for your research, please cite our paper as:

```
@article{zhang2024constrained,
  title={Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning},
  author={Zhang, Jing and Zhang, Chi and Wang, Wenjia and Jing, Bingyi},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```



## Acknowledgement

This codebase is built off of the official implementation of BEAR (https://github.com/rail-berkeley/d4rl_evaluations/tree/master/bear) ,rlkit (https://github.com/vitchyr/rlkit/) and Flow GAN (https://github.com/ermongroup/flow-gan). 
