# test_avoid.py

import time
import numpy as np

from gym_pybullet_drones.envs.AvoidAviary import AvoidAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# 如果安装了 stable-baselines3，就可以用它做一个简短的训练演示
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    USE_SB3 = True
except ImportError:
    USE_SB3 = False
    print("stable-baselines3 未安装，跳过训练示例，只跑随机测试。")

class RenderCallback(BaseCallback):
    """每一步都渲染 GUI（对于 PPO 训练时保持渲染）"""
    def _on_step(self) -> bool:
        return True

def random_play(env: AvoidAviary):
    """
    无限循环随机动作测试，直到手动退出（Ctrl+C）。
    """
    episode = 0
    try:
        while True:
            episode += 1
            obs, info = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                # 在动作空间 [-1,1]^3 上随机采样
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                # 以控制频率慢放
                time.sleep(1.0 / env.CTRL_FREQ)
            print(f"[Episode {episode}] 完成，距离终点 {info['distance_to_goal']:.3f} m")
    except KeyboardInterrupt:
        print("\n随机测试手动终止。")
    finally:
        env.close()

def rl_train(env: AvoidAviary, timesteps: int = 5000):
    """
    如果安装了 stable-baselines3，就用 PPO 训练一小段，
    并在 GUI 中观察网络输出动作的演化。
    """
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps, callback=RenderCallback())
    env.close()

if __name__ == "__main__":
    # 创建 AvoidAviary 环境，开启 GUI 并禁止录像
    env = AvoidAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=True,
        record=False,
        debug=True           # 打开 DEBUG，可以在 step() 看到原始动作打印
    )

    print("=== GUI 随机测试，按 Ctrl+C 退出 ===")
    random_play(env)