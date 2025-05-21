# ./examples/learn_Nav.py
import argparse
import os
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import wandb
from gym_pybullet_drones.utils.wandb_callback import WandbCallback

# 导入自定义环境
from gym_pybullet_drones.envs.NavRLAviary import NavRLAviary

# ------------------ 默认参数 ------------------
DEFAULT_OUTPUT_DIR       = 'nav_results'
DEFAULT_TOTAL_STEPS      = int(3e6)
DEFAULT_SAVE_FREQ        = 100_000
DEFAULT_WANDB_PROJECT    = 'nav-ppo'
DEFAULT_GUI              = False        # 默认评估/演示阶段是否开启 GUI
DEFAULT_STOP_REWARD      = 1500.0       # PPO 停训练的回报阈值
DEFAULT_LOG_INTERVAL     = 100          # 每隔 log_interval 个 timestep看到汇总
DEFAULT_EVAL_EPISODES    = 5            # evaluate_policy 时每次评估的回合数
DEFAULT_EVAL_FREQ        = 1000         # EvalCallback 的评估频率
DEFAULT_N_ENVS           = 10            # 并行环境数量

# ----------------------------------------------


class RewardPartLogger(BaseCallback):
    """
    每 step 从 info 里取出 r_vel…r_height，打到 wandb
    """
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        info0 = infos[0]
        # 提取五项子奖励
        keys = ("r_vel","r_ss","r_ds","r_smooth","r_height")
        metrics = {f"reward/{k}": info0.get(k, 0.0) for k in keys}
        wandb.log(metrics, step=self.num_timesteps)
        return True


def main(demo_gui: bool = DEFAULT_GUI,
         wandb_flag: bool = True,
         project: str = DEFAULT_WANDB_PROJECT,
         entity: str | None = None,
         run_name: str | None = None):
    # ============ 1. 初始化 wandb =================
    if wandb_flag:
        wandb.init(
            project=project,
            entity=entity,
            name=run_name or f"NavRL-{datetime.now().strftime('%m%d_%H%M%S')}",
            config={
                "total_steps": DEFAULT_TOTAL_STEPS,
                "save_freq": DEFAULT_SAVE_FREQ,
                "eval_freq": DEFAULT_EVAL_FREQ,
                "stop_reward": DEFAULT_STOP_REWARD,
                "log_interval": DEFAULT_LOG_INTERVAL,
                "eval_episodes": DEFAULT_EVAL_EPISODES,
                "env": "NavRLAviary",
                "algo": "PPO",
            },
            sync_tensorboard=True,
        )

    # ============ 2. 结果目录 =====================
    out_dir = os.path.join(DEFAULT_OUTPUT_DIR, datetime.now().strftime('%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)

    # ============ 3. 创建环境 =====================
    env_kwargs = dict(gui=False)  # 训练禁用 GUI 提速

    train_env = make_vec_env(NavRLAviary, n_envs=DEFAULT_N_ENVS, env_kwargs=env_kwargs)
    eval_env = Monitor(NavRLAviary(gui=False))


    # 打印空间信息 (debug)
    print('[DEBUG] Obs space', train_env.observation_space)
    print('[DEBUG] Act space', train_env.action_space)

    # ============ 4. 定义模型 & 回调 =============
    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=out_dir+'/tb')

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=DEFAULT_STOP_REWARD, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=out_dir,
                                 log_path=out_dir,
                                 eval_freq=DEFAULT_EVAL_FREQ,
                                 callback_on_new_best=stop_callback,
                                 deterministic=True,
                                 render=False,
                                 verbose=1)
    wandb_cb = (WandbCallback(save_freq=DEFAULT_SAVE_FREQ,
                              save_path=out_dir+'/wandb_ckpt',
                              verbose=1) if wandb_flag else None)

    callbacks = [eval_callback, RewardPartLogger()]
    if wandb_cb:
        callbacks.append(wandb_cb)

    # ============ 5. 训练 =========================
    model.learn(
        total_timesteps=DEFAULT_TOTAL_STEPS,
        callback=callbacks,
        log_interval=DEFAULT_LOG_INTERVAL,
    )
    # 保存模型并打印路径
    save_path = os.path.join(out_dir, 'final_model.zip')
    model.save(save_path)
    print(f"[INFO] 模型已保存至: {save_path}")

    # ============ 6. 评估和演示 ===================
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=DEFAULT_EVAL_EPISODES
    )
    eval_env.close()

    print(f"[INFO] Evaluate mean_reward={mean_reward:.2f} ± {std_reward:.2f}")
    if wandb_flag:
        wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})

    if demo_gui:
        test_env = NavRLAviary(gui=True)
        obs, _ = test_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            ep_r += reward
            test_env.render()
            done = terminated or truncated
        print(f"[INFO] 演示回合奖励: {ep_r:.2f}")
        test_env.close()

    if wandb_flag:
        wandb.finish()

# -------------------- CLI ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_gui',   default=True, type=bool, help='是否可视化测试')
    parser.add_argument('--wandb_flag', default=True, type=bool, help='启用 wandb')
    parser.add_argument('--run_name', default=None, type=str, help='wandb run 名称')
    parser.add_argument('--project', default=DEFAULT_WANDB_PROJECT, type=str)
    parser.add_argument('--entity',  default=None, type=str)
    args = parser.parse_args()
    main(demo_gui=args.demo_gui,
         wandb_flag = args.wandb_flag,
         project = args.project,
         entity = args.entity,
         run_name = args.run_name)