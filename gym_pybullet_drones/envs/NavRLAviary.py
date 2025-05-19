# ./envs/NavRLAviary.py

"""NavRL‑style 训练环境，基于 gym‑pybullet‑drones 的 BaseRLAviary。

主要特性
---------
* **连续 4 维动作**：v_x, v_y, v_z, ω_yaw，对应 ActionType.VEL
* **观测**：
  - S_int  (5)   ‑ 目标方向单位向量(3) + 与目标距离(1) + 无人机速度(1 在此按 V_r.xy 模长简化)
  - S_dyn  (N_DYN_OBS * 8)   ‑ 未启用，全部填 0；接口保留
  - S_stat (N_H * N_V)       ‑ 激光射线距离，以 body frame 表示
  - 最近 ACTION_BUFFER_SIZE 步动作（与父类保持一致）
* **奖励**：公式 (8)–(12) 全部实现，可通过 λ_i 进行加权

如需完整 NavRL 功能，只需替换 `_get_dynamic_obstacles()` 部分并根据实际场景
调整光线投射函数 `_cast_static_rays()` 即可。
"""

from __future__ import annotations

import math
import os
from typing import Tuple, List

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, DroneModel, Physics

# ======================== 全局 / 默认参数 ========================
DEFAULT_N_H                = 36      # 水平射线数量 (每 10° 一条)
DEFAULT_N_V                = 2       # 垂直平面数量 (俯仰角 0°, −15°)
DEFAULT_N_DYN_OBS          = 5       # 最近动态障碍数量上限
DEFAULT_DYN_FEATURE_DIM    = 8       # 每个动态障碍特征维度
DEFAULT_MAX_EPISODE_SEC    = 20      # 单集最长秒数
DEFAULT_MAX_VEL_MPS        = 3.0     # 最大速度 (m/s)
DEFAULT_GOAL_TOL_DIST      = 0.3     # 视为到达目标的距离阈值 (m)
DEFAULT_S_INT_DIM          = 5       # S_int 维度
DEFAULT_ACTION_DIM         = 4       # 动作维度 (VEL -> 4)
DEFAULT_SAMPLING_RANGE     = 25.0    # 50×50 m 场地的一半


# 奖励权重 λ_i
LAMBDA_VEL     = 1.0
LAMBDA_SS      = 1.0
LAMBDA_DS      = 1.0
LAMBDA_SMOOTH  = 0.2
LAMBDA_HEIGHT  = 0.1

# ===============================================================

class NavRLAviary(BaseRLAviary):
    """简化 NavRL 无人机导航环境。"""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 n_h: int = DEFAULT_N_H,
                 n_v: int = DEFAULT_N_V,
                 n_dyn_obs: int = DEFAULT_N_DYN_OBS,
                 goal_tol: float = DEFAULT_GOAL_TOL_DIST,
                 max_episode_sec: int = DEFAULT_MAX_EPISODE_SEC,
                 **base_kwargs):
        # 保存自定义参数
        self.N_H = n_h
        self.N_V = n_v
        self.N_D = n_dyn_obs
        self.goal_tol = goal_tol
        self.EPISODE_SEC = max_episode_sec

        # 每个 episode 随机生成起始/目标点时的采样边界 (正方形)
        self.SAMPLING_RANGE = DEFAULT_SAMPLING_RANGE

        # 用于奖励计算的上一步速度缓存
        self.prev_vel_world = np.zeros(3)

        self.P_s = np.zeros(3)             # 起点占位
        self.P_g = np.array([1., 0., 0.])  # 目标占位，避免零向量除以 0

        # 调用父类构造函数 (obs=KIN, act=VEL)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL,
                         **base_kwargs)

        # 预计算光线单位方向 (body frame)
        self._ray_directions_body = self._precompute_ray_dirs()

        # 用来存上一次 debug 文本的 id
        self._start_text_id = None
        self._goal_text_id  = None

    # ------------------------ Episode 管理 ------------------------

    def reset(self, seed: int | None = None, options=None):  # noqa: D401
        """重置环境：随机起点 Ps 与目标 Pg，并建立目标坐标系。"""
        # 1. 删除上一集的标记
        if self._start_text_id is not None:
            p.removeUserDebugItem(self._start_text_id, physicsClientId=self.CLIENT)
        if self._goal_text_id is not None:
            p.removeUserDebugItem(self._goal_text_id,  physicsClientId=self.CLIENT)

        obs, info = super().reset(seed=seed, options=options)

        # 取当前无人机位置作为 Ps
        state = self._getDroneStateVector(0)
        self.P_s = state[0:3].copy()

        # 随机采样目标 Pg
        dx, dy = np.random.uniform(-self.SAMPLING_RANGE, self.SAMPLING_RANGE, size=2)
        self.P_g = self.P_s + np.array([dx, dy, 0])  # 与起点同高

        # 预计算 goal frame 旋转矩阵 (body/world → goal)
        fg = self.P_g - self.P_s
        fg_xy_norm = np.linalg.norm(fg[0:2]) + 1e-6
        cos_yaw = fg[0] / fg_xy_norm
        sin_yaw = fg[1] / fg_xy_norm
        # 将 world 坐标绕 Z 轴旋转，使 x 轴指向目标
        self.R_W2G = np.array([[ cos_yaw, sin_yaw, 0],
                               [-sin_yaw, cos_yaw, 0],
                               [      0 ,      0 , 1]])

        # 重置速度缓存
        self.prev_vel_world = np.zeros(3)
        self.step_counter = 0

        p.addUserDebugText(
            text="S",
            textPosition=self.P_s.tolist(),
            textColorRGB=[0, 1, 0],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.CLIENT
        )

        p.addUserDebugText(
            text="G",
            textPosition=self.P_g.tolist(),
            textColorRGB=[1, 0, 0],
            textSize=1.5,
            lifeTime=0,
            physicsClientId=self.CLIENT
        )
        return self._computeObs(), info

    # ------------------------ Observation ------------------------

    def _precompute_ray_dirs(self) -> np.ndarray:
        """预先计算所有光线在机体坐标系下的单位方向向量。"""
        dirs: List[np.ndarray] = []
        for j in range(self.N_V):
            pitch_deg = -15.0 * j  # 0°, −15°, ...
            for i in range(self.N_H):
                yaw_deg = 360.0 * i / self.N_H
                dirs.append(self._euler_deg_to_unit(yaw_deg, pitch_deg))
        return np.stack(dirs, axis=0)  # (N_H*N_V, 3)

    @staticmethod
    def _euler_deg_to_unit(yaw_deg: float, pitch_deg: float) -> np.ndarray:
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        # 偏航 + 俯仰，仅绕 Z 再绕 Y
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x = cp * cy
        y = cp * sy
        z = -sp  # 俯视为负
        return np.array([x, y, z])

    def _cast_static_rays(self, pos_world: np.ndarray) -> np.ndarray:
        """在 world 坐标系下投射所有射线并返回距离 (m)。"""
        # 射线最大长度：20 m（可调）
        RAY_LEN = 20.0
        ray_from = np.repeat(pos_world[None, :], repeats=self.N_H * self.N_V, axis=0)
        ray_dirs_world = self._ray_directions_body  # body ≈ world (无滚俯假设)
        ray_to = ray_from + ray_dirs_world * RAY_LEN
        # 批量射线测试
        results = p.rayTestBatch(ray_from.tolist(), ray_to.tolist(), physicsClientId=self.CLIENT)
        dists = np.array([hit[2] * RAY_LEN for hit in results])  # hit[2] 是 hitFraction
        return dists  # shape (N_H * N_V,)

    def _get_dynamic_obstacles(self, pos_world: np.ndarray) -> Tuple[np.ndarray, int]:
        """获取最近 N_D 个动态障碍 (简化为 0 个，占位)。"""
        # TODO: 若需要，可在此扫描仿真中其他实体
        return np.zeros((self.N_D, 8)), 0  # (N_D, 特征维), 实际数量

    def _computeObs(self):
        """构造符合 NavRL 设计的观测向量。"""
        state = self._getDroneStateVector(0)
        P_r_W = state[0:3]
        V_r_W = state[10:13]

        # === S_int ===
        dir_to_goal = self.P_g - P_r_W
        dist_to_goal = np.linalg.norm(dir_to_goal) + 1e-6
        dir_unit = dir_to_goal / dist_to_goal
        S_int = np.hstack([dir_unit, dist_to_goal, V_r_W[0:1]])  # 5 维，速度简化为 v_x

        # === S_dyn (未实现) ===
        S_dyn, _ = self._get_dynamic_obstacles(P_r_W)          # (N_D, 8)
        S_dyn_flat = S_dyn.flatten()

        # === S_stat ===
        ray_dist = self._cast_static_rays(P_r_W)                # (N_H*N_V,)

        # === 上一步动作缓存 ===
        act_buf = np.hstack([self.action_buffer[i][0, :] for i in range(self.ACTION_BUFFER_SIZE)])

        obs_vec = np.hstack([S_int, S_dyn_flat, ray_dist, act_buf]).astype(np.float32)
        return obs_vec.reshape(1, -1)  # Gymnasium 多智能体接口期望 (num_drones, obs_dim)


    def _observationSpace(self):
        """
        根据 DEFAULT 参数计算观测维度：
          S_int + S_dyn + S_stat + 动作缓存
        """
        obs_dim = (
            DEFAULT_S_INT_DIM                  # S_int
            + self.N_D * DEFAULT_DYN_FEATURE_DIM  # S_dyn
            + self.N_H * self.N_V              # S_stat
            + self.ACTION_BUFFER_SIZE * DEFAULT_ACTION_DIM  # 动作缓存
        )
        low = -np.inf * np.ones((1, obs_dim), dtype=np.float32)
        high = np.inf * np.ones((1, obs_dim), dtype=np.float32)
        from gymnasium import spaces
        return spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------ Reward & Termination ------------------------

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        P_r_W = state[0:3]
        V_r_W = state[10:13]

        # r_vel
        dir_to_goal = self.P_g - P_r_W
        dist_to_goal = np.linalg.norm(dir_to_goal) + 1e-6
        dir_unit = dir_to_goal / dist_to_goal
        r_vel = float(np.dot(dir_unit, V_r_W))

        # r_ss
        ray_dist = self._cast_static_rays(P_r_W)
        ray_dist[ray_dist == 0] = 1e-3  # 避免 log(0)
        r_ss = float(np.mean(np.log(ray_dist)))

        # r_ds (未实现动态障碍)
        r_ds = 0.0

        # r_smooth
        r_smooth = -float(np.linalg.norm(V_r_W - self.prev_vel_world))
        self.prev_vel_world = V_r_W.copy()

        # r_height
        r_height = -float(min(abs(P_r_W[2] - self.P_s[2]), abs(P_r_W[2] - self.P_g[2])) ** 2)

        reward = (LAMBDA_VEL * r_vel + LAMBDA_SS * r_ss + LAMBDA_DS * r_ds +
                  LAMBDA_SMOOTH * r_smooth + LAMBDA_HEIGHT * r_height)
        return reward

    def _computeTerminated(self) -> bool:
        """
        terminated=True   → 任务本身结束（例如到达目标或碰撞）
        """
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.P_g - state[0:3])
        return dist < self.goal_tol  # 仅成功条件

    def _computeInfo(self):
        """返回与导航相关的实时信息，可用于调试或评估。"""
        state = self._getDroneStateVector(0)
        dist_to_goal = float(np.linalg.norm(self.P_g - state[0:3]))
        return {
            "step": int(self.step_counter),
            "distance_to_goal": dist_to_goal
        }

    def _computeTruncated(self):
        """超时截断：到达最大步数即结束。"""
        return self.step_counter >= self.EPISODE_SEC * self.CTRL_FREQ


    # ----------- PyBullet step 钩子 (父类 step 会调用) -----------

    def _postAction(self):
        self.step_counter += 1  # 追踪当前步数