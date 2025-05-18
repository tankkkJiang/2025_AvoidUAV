"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from gym_pybullet_drones.utils.wandb_callback import WandbCallback  # 自定义 wandb 回调

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_SAVE_FREQ = 10_000  # 每多少步保存一次模型

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'

# 默认动作类型：‘one_d_rpm’ 表示网络输出每个桨相对悬停转速的增量
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

# 默认多机环境中的无人机数量
DEFAULT_AGENTS = 2

# 默认是否启用多智能体（multiagent）模式，False 表示单机训练
DEFAULT_MA = False

# 根据运行环境不同选择训练步数
DEFAULT_TIMESTEPS_LOCAL = int(1e7)
DEFAULT_TIMESTEPS_REMOTE = int(1e2)


def run(multiagent=DEFAULT_MA,
            output_folder=DEFAULT_OUTPUT_FOLDER,
            gui=DEFAULT_GUI,
            plot=True,
            colab=DEFAULT_COLAB,
            record_video=DEFAULT_RECORD_VIDEO,
            local=True,
            wandb_flag=True,  # wandb 开关
            project='gpd-ppo',  # wandb project 名称
            entity=None,  # wandb entity 名称
            run_name=None  # wandb run 名称
            ):
    # 初始化 wandb
    if wandb_flag:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name or f"run-{datetime.now().strftime('%m%d_%H%M%S')}",
            config=dict(
                env='MultiHoverAviary' if multiagent else 'HoverAviary',
                obs=str(DEFAULT_OBS),
                act=str(DEFAULT_ACT),
                agents=DEFAULT_AGENTS if multiagent else 1,
                total_timesteps=DEFAULT_TIMESTEPS_LOCAL if local else DEFAULT_TIMESTEPS_REMOTE,
                save_freq=DEFAULT_SAVE_FREQ  # 将保存频率也记录到 config
            ),
            sync_tensorboard=True
        )

    # 生成结果保存目录
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # 创建训练和评估环境
    if not multiagent:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        train_env = make_vec_env(MultiHoverAviary,
                                 env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)

    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 467. if not multiagent else 920.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    # 开始训练，组合回调
    total_steps = DEFAULT_TIMESTEPS_LOCAL if local else DEFAULT_TIMESTEPS_REMOTE
    print(f"开始训练，总步数 = {total_steps}")
    wandb_cb = WandbCallback(save_freq=10000,
                             save_path=os.path.join(filename, 'wandb_ckpt'),
                             verbose=1) if wandb_flag else None

    callback_list = [eval_callback]
    if wandb_cb:
        callback_list.append(wandb_cb)

    model.learn(total_timesteps=total_steps,
                callback=callback_list,
                log_interval=100)

    # 保存最终模型
    model.save(filename+'/final_model.zip')
    print(filename)


    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if local:
        input("Press Enter to continue...")

    # if os.path.isfile(filename+'/final_model.zip'):
    #     path = filename+'/final_model.zip'
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
    # 记录测试指标到 wandb
    if wandb_flag:
        wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})


    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

    # 结束 wandb run
    if wandb_flag:
        wandb.finish()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    # 增加 wandb 参数
    parser.add_argument('--wandb_flag', default=True, type=str2bool, help='是否启用 wandb')
    parser.add_argument('--project', default='gpd-ppo', type=str, help='wandb project 名称')
    parser.add_argument('--entity', default=None, type=str, help='wandb entity/团队')
    parser.add_argument('--run_name', default=None, type=str, help='wandb run 名称')

    ARGS = parser.parse_args()

    run(**vars(ARGS))
