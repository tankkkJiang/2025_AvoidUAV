# test_navrl_gui.py

import os
import sys
import time
import numpy as np

from gym_pybullet_drones.envs.NavRLAviary import NavRLAviary
from gym_pybullet_drones.utils.enums import DroneModel

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    USE_SB3 = True
except ImportError:
    USE_SB3 = False
    print("stable-baselines3 未安装，跳过训练示例，只跑随机测试。")

class RenderCallback(BaseCallback):
    """在每一步都让 PyBullet GUI 刷新。"""
    def _on_step(self) -> bool:
        return True

def random_play(env: NavRLAviary):
    """
    无限循环随机动作测试，直到手动退出（Ctrl+C 或关闭窗口）。
    """
    try:
        while True:
            obs, info = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                # 控制步速，使视觉效果可见
                time.sleep(1.0 / env.CTRL_FREQ)
    except KeyboardInterrupt:
        print("\n随机测试已手动终止。")
    finally:
        env.close()

def rl_train(env: NavRLAviary, timesteps: int = 10000):
    """用 PPO 在 GUI 环境中训练一小段，并观察学习过程。"""
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=RenderCallback())
    env.close()

if __name__ == "__main__":
    # 创建环境 —— 注意传 gui=True
    env = NavRLAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=True,           # 打开 PyBullet GUI
        record=False,       # 如需录像可打开
        max_episode_sec=10  # 每集最长 10s（可按需调节）
    )

    print("=== 无限随机测试，按 Ctrl+C 可退出 ===")
    random_play(env)