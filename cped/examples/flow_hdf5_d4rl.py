from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, VAEPolicy
from rlkit.torch.sac.flow import FLOWTrainer
from rlkit.torch.networks import FlattenMlp,Actor,Critic
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.FlowGAN import RealNVP, NICE,Discriminator
import numpy as np

import h5py, argparse, os
import gym
import d4rl
import torch

def load_hdf5(dataset, replay_buffer, max_size,envname):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], max_size)

    _obs = all_obs[:N-1]
    _actions = all_act[:N-1]
    _next_obs = all_obs[1:]
    if 'antmaze' in envname:
       _rew =(np.expand_dims(dataset['rewards'][:N-1], 1) - 1)
    else:
       _rew = np.squeeze(dataset['rewards'][:N-1])
       _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
    _done = np.squeeze(dataset['terminals'][:N-1])
    _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

    max_length = 1000
    ctr = 0
    ## Only for MuJoCo environments
    ## Handle the condition when terminal is not True and trajectory ends due to a timeout
    for idx in range(_obs.shape[0]):
        if ctr  >= max_length - 1:
            ctr = 0
        else:
            replay_buffer.add_sample_only(_obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
            ctr += 1
            if _done[idx][0]:
                ctr = 0
    ###

    print (replay_buffer._size, replay_buffer._terminals.shape)


def experiment(variant):


    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    max_action = float(eval_env.action_space.high[0])

    M = variant['layer_size']
    qf1 = Critic(obs_dim, action_dim)
    qf2 = Critic(obs_dim, action_dim)
    target_qf1 = Critic(obs_dim, action_dim)
    target_qf2 = Critic(obs_dim, action_dim)
    policy = Actor(obs_dim, action_dim, max_action)
    target_policy=Actor(obs_dim, action_dim, max_action)
    
    dis = Discriminator(
        input_size=obs_dim + action_dim,
        output_size=obs_dim + action_dim,
        hidden_sizes=[2*(obs_dim+action_dim), 4*(obs_dim+action_dim),],
    )

    if variant['model_type']=="real_nvp":
        flowmodel = RealNVP(
            xdim=obs_dim + action_dim,
            mask_type='checkerboard0',
            no_of_layers=4,
        )
    elif variant['model_type']=="nice":
        flowmodel = NICE(
            xdim=obs_dim + action_dim,
            mask_type='checkerboard0',
            no_of_layers=4,
        )
    else:
        raise NameError("model_type undefined: {}".format(variant['algorithm']))
    eval_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer, max_size=variant['replay_buffer_size'],envname=variant['env_name'])
    
    trainer = FLOWTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        flow=flowmodel,
        discriminator=dis,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='BEAR-runs')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--mmd_sigma', default=50, type=float)
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--target_mmd_thresh', default=0.05, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--model_type', default="nice", type=str)
    parser.add_argument('--flowgan_train_epoch', default=50, type=int)
    args = parser.parse_args()
    
    if('antmaze' in args.env):
       variant = dict(
        algorithm="BEAR",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=True,
        env_name=args.env,
        # Flow specific algprithm
        model_type=args.model_type,
        flowgan_train_epoch=args.flowgan_train_epoch,
        
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples
            ),
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            batch_size=256,

            #flow specific
            prior="logistic",
            f_div="wgan",
            model_type=args.model_type,
            flowgan_train_epoch=args.flowgan_train_epoch,
            env_name=args.env
            ),
        )
    else:
        variant = dict(
        algorithm="BEAR",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=True,
        env_name=args.env,
        # Flow specific algprithm
        model_type=args.model_type,
        flowgan_train_epoch=args.flowgan_train_epoch,
        
        algorithm_kwargs=dict(
            num_epochs=args.flowgan_train_epoch+1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples
            ),
        trainer_kwargs=dict(
            discount=0.99,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            batch_size=256,

            #flow specific
            prior="logistic",
            f_div="wgan",
            model_type=args.model_type,
            flowgan_train_epoch=args.flowgan_train_epoch,
            env_name=args.env
            ),
        )
        

    
    
#with torch.autograd.detect_anomaly():
    np.random.seed()
    rand = np.random.randint(0, 100000)
    setup_logger(os.path.join('CPED_launch', str(rand)), variant=variant, base_log_dir='./cped')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    
    experiment(variant)
