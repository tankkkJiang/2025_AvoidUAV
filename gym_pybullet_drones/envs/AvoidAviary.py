# ./envs/AvoidAviary.py

from __future__ import annotations

import math
import os
from typing import Tuple, List

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ======================== 全局 / 默认参数 ========================
DEFAULT_N_H                = 36      # 水平射线数量
DEFAULT_N_V                = 2       # 垂直平面数量
DEFAULT_N_DYN_OBS          = 5       # 动态障碍上限
DEFAULT_DYN_FEATURE_DIM    = 8       # 动态障碍特征维
DEFAULT_MAX_EPISODE_SEC    = 100     # 最长秒数
DEFAULT_CTRL_FREQ          = 48      # 控制频率 (Hz) = step 频率
DEFAULT_GOAL_TOL_DIST      = 0.3     # 成功阈值 (m)
DEFAULT_S_INT_DIM          = 5
DEFAULT_SAMPLING_RANGE     = 5.0     # 起点采样半边长 (m)
DEFAULT_DEBUG              = False

# -------- 动作相关 --------
DEFAULT_ACTION_DIM         = 3                   # Δx, Δy, Δz
DEFAULT_MAX_DELTA_M        = 0.02                # Δ 最大值 (m) ← 可调
DEFAULT_MAX_YAW_RATE       = math.pi / 3         # 保留偏航速率限制
DEFAULT_SPEED_RATIO        = 1.0                 # PID 速度比例

# 观测用静态障碍
DEFAULT_OBSTACLE_URDF      = "cube.urdf"
DEFAULT_SCENARIO           = "simple"
DEFAULT_ENABLE_STATIC_OBS  = True
DEFAULT_NUM_STATIC_OBS     = 10

# 奖励权重
LAMBDA_VEL     = 1.0
LAMBDA_SS      = 1.0
LAMBDA_DS      = 1.0
LAMBDA_SMOOTH  = 0.1
LAMBDA_HEIGHT  = 0.1
# ===============================================================


class AvoidAviary(BaseRLAviary):
    """基于 Δx,Δy,Δz 动作的单机导航‑避障环境。"""
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 n_h: int = DEFAULT_N_H,
                 n_v: int = DEFAULT_N_V,
                 n_dyn_obs: int = DEFAULT_N_DYN_OBS,
                 goal_tol: float = DEFAULT_GOAL_TOL_DIST,
                 max_episode_sec: int = DEFAULT_MAX_EPISODE_SEC,
                 ctrl_freq: int = DEFAULT_CTRL_FREQ,
                 enable_static_obs: bool = DEFAULT_ENABLE_STATIC_OBS,
                 num_static_obs: int = DEFAULT_NUM_STATIC_OBS,
                 debug: bool = DEFAULT_DEBUG,
                 scenario: str = DEFAULT_SCENARIO,
                 **base_kwargs):
        # ------------ 保存自定义参数 ------------
        self.N_H           = n_h
        self.N_V           = n_v
        self.N_D           = n_dyn_obs
        self.goal_tol      = goal_tol
        self.EPISODE_SEC   = max_episode_sec
        self.CTRL_FREQ     = ctrl_freq
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ
        self.DEBUG         = debug
        self.MAX_STEPS     = self.EPISODE_SEC * self.CTRL_FREQ
        self.SCENARIO      = scenario
        self.SAMPLING_RANGE = DEFAULT_SAMPLING_RANGE

        # 用于奖励计算的上一步速度缓存
        self.prev_vel_world = np.zeros(3)

        # 起点占位；目标占位，避免零向量除以 0
        self.P_s = np.zeros(3)
        self.P_g = np.array([1., 0., 0.])

        # 静态障碍
        self._static_obstacle_ids: List[int] = []
        self.enable_static_obs = enable_static_obs
        self.num_static_obs = num_static_obs

        # 用来存放当前 step 各子奖励，用于分解奖励。
        self._reward_parts: dict = {
            "r_vel": 0.0,
            "r_ss": 0.0,
            "r_ds": 0.0,
            "r_smooth": 0.0,
            "r_height": 0.0,
        }

        # 把初始高度抬到 1 m 左右
        init_xyzs = np.array([[0.0, 0.0, 1.0]])  # (num_drones,3)
        base_kwargs.setdefault("initial_xyzs", init_xyzs)

        # 调用父类构造函数 (obs=KIN, act=VEL)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL,
                         ctrl_freq=self.CTRL_FREQ,
                         **base_kwargs)

        # 创建每架机对应的 DSLPIDControl
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X)
                         for _ in range(num_drones)]
        else:
            raise RuntimeError("NavRLAviary 目前只在 Crazyflie 上测试过。")

        # 预计算光线单位方向 (body frame)
        self._ray_directions_body = self._precompute_ray_dirs()

        # 用来存上一次 debug 文本的 id
        self._start_text_id = None
        self._goal_text_id  = None

        # 观测需要的动作缓存
        self.action_buffer.clear()
        zero_act = np.zeros((self.NUM_DRONES, DEFAULT_ACTION_DIM),
                            dtype=np.float32)
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(zero_act)


        if self.DEBUG:
            # 打印动作空间和观测空间的上下界，确认范围是否合理
            print(f"[DEBUG] action_space: {self.action_space}")
            print(f"[DEBUG] observation_space: {self.observation_space}")

    # ------------------------ Episode 管理 ------------------------
    def step(self, action):
        # 剪切到 [-1,1] 并写入 ring‑buffer（供观测使用）
        hat_delta = np.clip(action, -1., 1.).astype(np.float32)
        self.action_buffer.append(hat_delta.copy())
        self.last_highlevel_action = hat_delta.copy()

        if self.DEBUG:
            interval = max(1, self.MAX_STEPS // 10)
            if self.step_counter % interval == 0:
                # 原始动作 a ∈ [-1,1]
                print(f"[DEBUG] Step {self.step_counter:4d} ── ACTION(-1,1) ── "
                      f"{action.reshape(-1)}")
                # 映射到 Δpos ∈ [-DEFAULT_MAX_DELTA_M,DEFAULT_MAX_DELTA_M]
                delta_pos = hat_delta * DEFAULT_MAX_DELTA_M
                print(f"[DEBUG] Step {self.step_counter:4d} ── ΔPOS(m)    ── "
                      f"{delta_pos.reshape(-1)}")


        obs, reward, terminated, truncated, info = super().step(action)


        if self.DEBUG and (terminated or truncated):
            reason = "GOAL" if terminated else "TIMEOUT"
            print(f"[EPISODE END] reason={reason}  steps={self.step_counter}  dist={info['distance_to_goal']:.2f}")
            print(f"[DEBUG] EPISODE_SEC={self.EPISODE_SEC}, CTRL_FREQ={self.CTRL_FREQ}, MAX_STEPS={self.MAX_STEPS}")

        return obs, reward, terminated, truncated, info


    def reset(self, seed: int | None = None, options=None):  # noqa: D401
        """重置环境：随机起点 Ps 与目标 Pg，并建立目标坐标系。"""
        # 删除上一集 episode 的标记
        if self._start_text_id is not None:
            p.removeUserDebugItem(self._start_text_id, physicsClientId=self.CLIENT)
        if self._goal_text_id is not None:
            p.removeUserDebugItem(self._goal_text_id,  physicsClientId=self.CLIENT)
        # 删除旧的静态障碍
        if self.enable_static_obs:
            for oid in self._static_obstacle_ids:
                p.removeBody(oid, physicsClientId=self.CLIENT)
            self._static_obstacle_ids.clear()

        self.action_buffer.clear()
        # 一次性生成 (num_buffer, num_drones, action_dim) 的全零
        zero_act = np.zeros((self.NUM_DRONES, DEFAULT_ACTION_DIM), dtype=np.float32)
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(zero_act)
        self.last_highlevel_action = np.zeros((self.NUM_DRONES, DEFAULT_ACTION_DIM), dtype=np.float32)

        # 父类 reset
        obs, info = super().reset(seed=seed, options=options)

        # 取当前无人机位置作为 Ps
        state = self._getDroneStateVector(0)
        self.P_s = state[0:3].copy()
        # 随机采样目标 Pg
        dx, dy = np.random.uniform(-self.SAMPLING_RANGE, self.SAMPLING_RANGE, size=2)
        self.P_g = self.P_s + np.array([dx, dy, 0.0])  # 与起点同高
        init_dist = np.linalg.norm(self.P_g - self.P_s)
        if self.DEBUG:
            print(f"\n[RESET] P_s = {self.P_s},  P_g = {self.P_g}")
            print(f"[RESET] Initial distance = {init_dist:.3f} m")

        # 添加随机静态障碍
        if self.enable_static_obs:
            self._add_static_obstacles()

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

    # ------------------------ Action space -----------------------
    def _actionSpace(self):
        from gymnasium import spaces
        low = -np.ones((1, DEFAULT_ACTION_DIM), dtype=np.float32)
        high = np.ones((1, DEFAULT_ACTION_DIM), dtype=np.float32)
        return spaces.Box(low=low, high=high, dtype=np.float32)

    #  动作 → 4 电机 RPM
    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """把 Δx,Δy,Δz 映射为局部目标点，再用 PID 求 RPM。"""
        rpm = np.zeros((self.NUM_DRONES, 4))
        # 把 [-1,1] 映射到 [-ΔMAX, ΔMAX]
        delta_pos = action * DEFAULT_MAX_DELTA_M  # (N,3)
        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            cur_pos   = state[0:3]
            cur_quat  = state[3:7]
            cur_vel   = state[10:13]
            cur_omega = state[13:16]

            # 目标点 = 当前 + Δ
            target_pos = cur_pos + delta_pos[k]
            target_yaw = state[9]  # 不改变偏航
            target_vel = np.zeros(3, dtype=np.float32)

            # DSLPIDControl → RPM
            rpm[k, :], _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_omega,
                target_pos=target_pos,
                target_rpy=np.array([0., 0., target_yaw]),
                target_vel=target_vel
            )
        return rpm

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

        self._reward_parts = {  # <— 新增
            "r_vel": r_vel,
            "r_ss": r_ss,
            "r_ds": r_ds,
            "r_smooth": r_smooth,
            "r_height": r_height,
        }

        return (LAMBDA_VEL * r_vel
            + LAMBDA_SS * r_ss
            + LAMBDA_DS * r_ds
            + LAMBDA_SMOOTH * r_smooth
            + LAMBDA_HEIGHT * r_height)

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
            "distance_to_goal": dist_to_goal,
            **self._reward_parts
        }

    def _computeTruncated(self):
        """超时截断：到达最大步数即结束。"""
        return self.step_counter >= self.MAX_STEPS


    # ----------- PyBullet step 钩子 (父类 step 会调用) -----------

    def _postAction(self):
        self.step_counter += 1  # 追踪当前步数


    # ----------- 辅助方法 -----------
    def _add_static_obstacles(self):
        """根据不同场景在环境中添加静态障碍物。"""
        if not self.enable_static_obs:
            return

        match self.SCENARIO:
            case "simple":
                # 在起点 Ps 与终点 Pg 的中点放一个方块
                mid = (self.P_s + self.P_g) / 2.0
                oid = p.loadURDF(
                    DEFAULT_OBSTACLE_URDF,
                    basePosition=[mid[0], mid[1], 0.5],
                    globalScaling=1.0,
                    physicsClientId=self.CLIENT
                )
                self._static_obstacle_ids.append(oid)
                if self.DEBUG:
                    print(f"[DEBUG] simple: placed 1 box at midpoint {mid.tolist()}")

            case "random":
                # 随机散布 num_static_obs 个方块
                for _ in range(self.num_static_obs):
                    dx, dy = np.random.uniform(-self.SAMPLING_RANGE, self.SAMPLING_RANGE, size=2)
                    pos = [self.P_s[0] + dx, self.P_s[1] + dy, 0.5]
                    oid = p.loadURDF(
                        DEFAULT_OBSTACLE_URDF,
                        basePosition=pos,
                        globalScaling=1.0,
                        physicsClientId=self.CLIENT
                    )
                    self._static_obstacle_ids.append(oid)
                if self.DEBUG:
                    print(f"[DEBUG] random: placed {self.num_static_obs} boxes")

            case "circle":
                # 在起点周围围成一个圆环
                for i in range(self.num_static_obs):
                    theta = 2*math.pi * i / self.num_static_obs
                    r = self.SAMPLING_RANGE * 0.5
                    pos = [
                        self.P_s[0] + r*math.cos(theta),
                        self.P_s[1] + r*math.sin(theta),
                        0.5
                    ]
                    oid = p.loadURDF(
                        DEFAULT_OBSTACLE_URDF,
                        basePosition=pos,
                        globalScaling=1.0,
                        physicsClientId=self.CLIENT
                    )
                    self._static_obstacle_ids.append(oid)
                if self.DEBUG:
                    print(f"[DEBUG] circle: placed {self.num_static_obs} boxes in a ring")

            case _:
                # 未知场景：不放障碍
                if self.DEBUG:
                    print(f"[DEBUG] unknown scenario '{self.SCENARIO}': no obstacles added")