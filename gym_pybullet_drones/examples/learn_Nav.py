# ============================= learn_Nav.py ==============================


"""训练 NavRLAviary 环境的示例脚本（PPO）。

运行示例
--------
$ python learn_Nav.py  # 本地 GUI 训练
$ python learn_Nav.py --local False --gui False  # 远程无 GUI 简易测试
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import wandb
from gym_pybullet_drones.utils.wandb_callback import WandbCallback

# 导入自定义环境
from gym_pybullet_drones.envs.NavRLAviary import NavRLAviary

# ------------------ 默认参数 ------------------
DEFAULT_OUTPUT_DIR    = 'nav_results'
DEFAULT_TOTAL_STEPS   = int(3e6)
DEFAULT_SAVE_FREQ     = 10_000
DEFAULT_WANDB_PROJECT = 'nav-ppo'
DEFAULT_GUI           = True   # 默认评估/演示阶段是否开启 GUI

# ----------------------------------------------

def main(local: bool = True,
         demo_gui: bool = DEFAULT_GUI,
         wandb_flag: bool = True,
         project: str = DEFAULT_WANDB_PROJECT,
         entity: str | None = None,
         run_name: str | None = None):
    # ============ 1. 初始化 wandb =================
    if wandb_flag:
        wandb.init(project=project,
                   entity=entity,
                   name=run_name or f"NavRL-{datetime.now().strftime('%m%d_%H%M%S')}",
                   config=dict(total_steps=DEFAULT_TOTAL_STEPS,
                                save_freq=DEFAULT_SAVE_FREQ),
                   sync_tensorboard=True)

    # ============ 2. 结果目录 =====================
    out_dir = os.path.join(DEFAULT_OUTPUT_DIR, datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    # ============ 3. 创建环境 =====================
    env_kwargs = dict(gui=False)  # 训练禁用 GUI 提速

    train_env = make_vec_env(NavRLAviary, n_envs=1, env_kwargs=env_kwargs)
    eval_env  = NavRLAviary(gui=False)

    # 打印空间信息 (debug)
    print('[DEBUG] Obs space', train_env.observation_space)
    print('[DEBUG] Act space', train_env.action_space)

    # ============ 4. 定义模型 & 回调 =============
    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=out_dir+'/tb')

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=150.0, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=out_dir,
                                 log_path=out_dir,
                                 eval_freq=5000,
                                 callback_on_new_best=stop_callback,
                                 deterministic=True,
                                 render=False)
    wandb_cb = (WandbCallback(save_freq=DEFAULT_SAVE_FREQ,
                              save_path=out_dir+'/wandb_ckpt',
                              verbose=1) if wandb_flag else None)

    callbacks = [eval_callback]
    if wandb_cb:
        callbacks.append(wandb_cb)

    # ============ 5. 训练 =========================
    model.learn(total_timesteps=DEFAULT_TOTAL_STEPS, callback=callbacks)
    # 保存模型并打印路径
    save_path = os.path.join(out_dir, 'final_model.zip')
    model.save(save_path)
    print(f"[INFO] 模型已保存至: {save_path}")

    # ============ 6. 评估和演示 ===================
    test_env = NavRLAviary(gui=demo_gui)
    while True:
        obs, _ = test_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            ep_reward += reward
            if demo_gui:
                test_env.render()
            done = terminated or truncated

        print(f"[INFO] 本次演示回合奖励: {ep_reward:.2f}")
        if wandb_flag:
            wandb.log({'eval/episode_reward': ep_reward})

        cmd = input("按回车键再演示一次，输入 'exit' 并回车退出: ")
        if cmd.strip().lower() == 'exit':
            break

    test_env.close()
    if wandb_flag:
        wandb.finish()

# -------------------- CLI ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', default=True, type=bool, help='本地长训练')
    parser.add_argument('--demo_gui',   default=True, type=bool, help='是否可视化测试')
    parser.add_argument('--wandb', default=True, type=bool, help='启用 wandb')
    parser.add_argument('--project', default=DEFAULT_WANDB_PROJECT, type=str)
    parser.add_argument('--entity',  default=None, type=str)
    args = parser.parse_args()
    main(local=args.local, demo_gui=args.demo_gui, wandb_flag=args.wandb, project=args.project, entity=args.entity)
