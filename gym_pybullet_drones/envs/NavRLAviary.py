# ./envs/NavRLAviary.py

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
DEFAULT_N_H                = 36      # 水平射线数量 (每 10° 一条)
DEFAULT_N_V                = 2       # 垂直平面数量 (俯仰角 0°, −15°)
DEFAULT_N_DYN_OBS          = 5       # 最近动态障碍数量上限
DEFAULT_DYN_FEATURE_DIM    = 8       # 每个动态障碍特征维度
DEFAULT_MAX_EPISODE_SEC    = 50      # 单集最长秒数
DEFAULT_CTRL_FREQ          = 60      # 每秒控制步数 (VeloctyAviary.ctrl_freq = 48)
# DEFAULT_MAX_STEPS          = DEFAULT_MAX_EPISODE_SEC * DEFAULT_CTRL_FREQ
DEFAULT_ACTION_HZ          = 60       # RL 每秒给几次动作，最好小于CTRL
DEFAULT_ACTION_REPEAT = DEFAULT_CTRL_FREQ // DEFAULT_ACTION_HZ
DEFAULT_GOAL_TOL_DIST      = 0.3     # 视为到达目标的距离阈值 (m)
DEFAULT_S_INT_DIM          = 7       # S_int 维度
DEFAULT_SAMPLING_RANGE     = 10.0     # 50×50 m 场地的一半
DEFAULT_DEBUG              = False   # 方便检查gui并打印episode结束原因
g  = 9.81                               # m/s²

# 动作缩放
DEFAULT_ACTION_DIM         = 3                       # 动作维度 (VEL -> 4)
DEFAULT_ACTION_PARAM_DIM   = DEFAULT_ACTION_DIM * 2  # 输出 α,β 各 DEFAULT_ACTION_DIM 个，共 2*DEFAULT_ACTION_DIM 维
DEFAULT_DETERMINISTIC      = False                   # 如果 True：部署阶段用 Beta 均值；False：训练阶段随机采样
DEFAULT_MAX_VEL_MPS        = 5                    # xy最大速度，注意 max_speed_kmh 30.000000
DEFAULT_MAX_VEL_Z          = 0                     # 垂直最大速度
DEFAULT_SPEED_RATIO        = 1                       # φ_speed，决定速度幅值的固定系数 (0~1)

# 静态障碍参数
DEFAULT_OBSTACLE_URDF = "cube.urdf"
DEFAULT_SCENARIO              = "simple"   # 可选 "random" | "simple" | "circle"
DEFAULT_ENABLE_STATIC_OBS     = True       # 是否启用随机静态障碍物
DEFAULT_NUM_STATIC_OBS        = 5         # 默认静态障碍物个数
COLLISION_DISTANCE_THRESH     = 0.05       # 5cm 以内即视为碰撞

# 奖励权重 λ_i
LAMBDA_VEL     = 10.0
LAMBDA_SS      = 1.0
LAMBDA_DS      = 1.0
LAMBDA_SMOOTH  = 0.1
LAMBDA_HEIGHT  = 0.1
COLLISION_PENALTY = -5.0      # 碰撞惩罚，大负值

# 观测
RAY_LEN               = 20.0     # 所有射线的最大长度 (m)
RAY_COLLISION_THRESH  = 2     # ＜ 此距离则触发碰撞(根据需求可调)
VIS_RAY_DEBUG         = True     # 打开/关闭 GUI 射线可视化

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
                 ctrl_freq: int = DEFAULT_CTRL_FREQ,
                 enable_static_obs: bool = DEFAULT_ENABLE_STATIC_OBS,
                 num_static_obs: int = DEFAULT_NUM_STATIC_OBS,
                 debug: bool = DEFAULT_DEBUG,
                 scenario: str = DEFAULT_SCENARIO,
                 action_repeat: int = DEFAULT_ACTION_REPEAT,
                 **base_kwargs):
        # 保存自定义参数
        self.N_H = n_h
        self.N_V = n_v
        self.N_D = n_dyn_obs
        self.goal_tol = goal_tol
        self.EPISODE_SEC = max_episode_sec
        self.CTRL_FREQ = ctrl_freq
        self.CTRL_TIMESTEP = 1 / self.CTRL_FREQ
        self.DEBUG = debug
        self.MAX_STEPS = self.EPISODE_SEC * self.CTRL_FREQ
        self.SCENARIO = scenario  # 场景类型
        self.ACTION_REPEAT = max(1, action_repeat)
        self.deterministic = DEFAULT_DETERMINISTIC
        self.collision_penalty = COLLISION_PENALTY


        # 每个 episode 随机生成起始/目标点时的采样边界 (正方形)
        self.SAMPLING_RANGE = DEFAULT_SAMPLING_RANGE

        self.collision = False  # 碰撞标志

        # 占位
        # 用于奖励计算的上一步速度缓存
        self.prev_vel_goal = np.zeros(3)
        # 起点占位；目标占位，避免零向量除以 0
        self.P_s = np.zeros(3)
        self.P_g = np.array([1., 0., 0.])
        self.R_W2G = np.eye(3)
        # 固定朝向用的 yaw（rad），会在 reset() 里覆盖
        self.fixed_target_yaw = 0.0
        self._ray_vis_ids: List[int] = []  # 保存上一帧 debug line 的 id
        self._horizontal_idx = np.arange(self.N_H)  # 俯仰角=0° 的索引

        # 静态障碍
        self._static_obstacle_ids: List[int] = []
        self.enable_static_obs = enable_static_obs
        self.num_static_obs = num_static_obs

        # 用来存放当前 step 各子奖励
        self._reward_parts: dict = {
            "r_vel": 0.0,
            "r_ss": 0.0,
            "r_ds": 0.0,
            "r_smooth": 0.0,
            "r_height": 0.0,
        }

        # 把初始高度抬高，第三维
        init_xyzs = np.array([[0.0, 0.0, 0.5]])  # (num_drones,3)
        base_kwargs.setdefault("initial_xyzs", init_xyzs)

        # 调用父类构造函数 (obs=KIN, act=VEL)
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         obs=ObservationType.KIN,
                         act=ActionType.VEL,
                         ctrl_freq=self.CTRL_FREQ,
                         **base_kwargs)


        self._drone_id = self.DRONE_IDS[0]

        # Beta 分布形状参数 (α, β)，对每个动作维度均相同
        self._beta_alpha = np.ones(DEFAULT_ACTION_DIM, dtype=np.float32) * 2.0
        self._beta_beta = np.ones(DEFAULT_ACTION_DIM, dtype=np.float32) * 2.0

        # 预计算光线单位方向 (body frame)
        self._ray_directions_body = self._precompute_ray_dirs()

        # 用来存上一次 debug 文本的 id
        self._start_text_id = None
        self._goal_text_id  = None


        if self.DEBUG:
            # 打印动作空间和观测空间的上下界，确认范围是否合理
            print(f"[DEBUG] action_space: {self.action_space}")
            print(f"[DEBUG] observation_space: {self.observation_space}")

    # ------------------------ Episode 管理 ------------------------
    def step(self, action):
        if self.step_counter % self.ACTION_REPEAT == 0:
            # 真的用到新动作；存进 ring buffer（用于观测）
            alpha = np.clip(action[:, :DEFAULT_ACTION_DIM], 1e-3, None)
            beta = np.clip(action[:, DEFAULT_ACTION_DIM:], 1e-3, None)
            if self.deterministic:
                u = alpha / (alpha + beta)  # Beta 均值
            else:
                u = np.random.beta(alpha, beta)  # 训练时随机采样
            # 映射到 [-1,1]
            hat_V = (2.0 * u - 1.0).astype(np.float32)
            self.action_buffer.append(hat_V.copy())
            self.last_highlevel_action = hat_V.copy()
        else:
            # 用上一次动作
            hat_V = self.last_highlevel_action
        # 将映射后速度当成“raw”动作，传给父类去算 RPM
        action = hat_V

        if self.DEBUG:
            interval = max(1, self.MAX_STEPS // 10)
            if self.step_counter % interval == 0:
                # 原始动作（High-level RL 输出）
                print(f"[DEBUG] Step {self.step_counter:4d} ── ACTION(raw) ── {np.array(action).reshape(-1)}")


        obs, reward, terminated, truncated, info = super().step(action)

        collided = self._check_collision()  # 里面会根据 DEBUG 打印最小距离
        if collided:
            self.collision = True


        if self.DEBUG and (terminated or truncated):
            reason = "GOAL" if terminated else "TIMEOUT"
            print(f"[EPISODE END] reason={reason}  steps={self.step_counter}  dist={info['distance_to_goal']:.2f}")
            print(f"[DEBUG] EPISODE_SEC={self.EPISODE_SEC}, CTRL_FREQ={self.CTRL_FREQ}, MAX_STEPS={self.MAX_STEPS}")
            # 经过 _preprocessAction -> clipped_action 后的 RPM
            # BaseAviary 会把 last_clipped_action 设为本步最终 RPM
            print(f"[DEBUG] RPM(applied)    ── {self.last_clipped_action[0].round(1)}")
            # 当前机体速度
            st = self._getDroneStateVector(0)
            vel = st[10:13]
            print(f"[DEBUG] VEL(current)    ── {vel.round(3)}")

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

        self.fixed_target_yaw = math.atan2(fg[1], fg[0])

        # 重置速度缓存
        self.prev_vel_goal = np.zeros(3)
        self.step_counter = 0
        self.collision = False

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

        # self._draw_rays(pos_world, dists)
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

        # 世界→目标坐标系下的位置差与速度
        delta_W = self.P_g - P_r_W  # world 下的到目标向量
        dist_to_goal = np.linalg.norm(delta_W) + 1e-6
        delta_G = self.R_W2G @ delta_W  # Goal Frame 下
        dir_unit_G = delta_G / dist_to_goal  # 方向单位向量
        V_r_G = self.R_W2G @ V_r_W  # 速度也转换到 Goal Frame

        # === S_int (Goal Frame) ===
        # [dir_x, dir_y, dir_z, distance, speed_along_goal]
        S_int = np.hstack([dir_unit_G, dist_to_goal, V_r_G])


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
        lo = np.full((1, DEFAULT_ACTION_PARAM_DIM), 1e-3, dtype=np.float32)
        hi = np.full((1, DEFAULT_ACTION_PARAM_DIM), 10.0, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    # ------------------------ Action → RPM ------------------------
    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        1. 把 [-1,1]^3 → 真实速度 v_des (m/s)
        2. 直接用 resetBaseVelocity 注入线速度
        3. 返回全 0 RPM，桨叶不再产生推力
        """
        # 目前只支持单架机
        k = 0
        v_hat = action[k, 0:DEFAULT_ACTION_DIM]          # [-1,1]
        v_des = np.array([
            v_hat[0] * DEFAULT_MAX_VEL_MPS,
            v_hat[1] * DEFAULT_MAX_VEL_MPS,
            v_hat[2] * DEFAULT_MAX_VEL_Z + g * self.CTRL_TIMESTEP
        ], dtype=np.float32) * DEFAULT_SPEED_RATIO       # (3,)

        state = self._getDroneStateVector(k)

        # -------- DEBUG 打印当前/目标速度 -------------
        if self.DEBUG:
            interval = max(1, self.MAX_STEPS // 10)
            if self.step_counter % interval == 0:
                cur_vel = state[10:13]
                cur_spd = np.linalg.norm(cur_vel)
                tgt_spd = np.linalg.norm(v_des)
                print(f"Step {self.step_counter:4d} - "
                      f"[VEL] now {cur_vel.round(3)} |{cur_spd:.3f} m/s  "
                      f"→  target {v_des.round(3)} |{tgt_spd:.3f} m/s")

        # -------- 直接写入线速度 ----------------------
        p.resetBaseVelocity(self._drone_id,
                            linearVelocity=v_des.tolist(),
                            angularVelocity=[0, 0, 0],
                            physicsClientId=self.CLIENT)

        # 返回零转速，避免多余力矩
        return np.zeros((self.NUM_DRONES, 4), dtype=np.float32)

    # ------------------------ Reward & Termination ------------------------

    def _computeReward(self):
        if self.collision:
            # 仍然记录子奖励供 info 输出
            self._reward_parts = {"r_vel": 0, "r_ss": 0, "r_ds": 0, "r_smooth": 0, "r_height": 0}
            return self.collision_penalty

        state = self._getDroneStateVector(0)
        P_r_W = state[0:3]
        V_r_W = state[10:13]

        # 世界→目标坐标系
        delta_W = self.P_g - P_r_W
        dist_to_goal = np.linalg.norm(delta_W) + 1e-6
        delta_G = self.R_W2G @ delta_W
        V_r_G = self.R_W2G @ V_r_W

        # r_vel：Goal Frame 下 x 方向的速度
        r_vel = float(V_r_G[0])

        # r_ss
        ray_dist = self._cast_static_rays(P_r_W)
        ray_dist[ray_dist == 0] = 1e-3  # 避免 log(0)
        r_ss = float(np.mean(np.log(ray_dist)))

        # r_ds (未实现动态障碍)
        r_ds = 0.0

        # r_smooth
        r_smooth = -float(np.linalg.norm(V_r_G - self.prev_vel_goal))
        self.prev_vel_goal = V_r_G.copy()

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
        # 碰撞直接终止
        if self.collision:
            return True

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
    def _applyMotorAction(self, rpm: np.ndarray):
        """覆盖父类：不施加任何推力/力矩."""
        pass


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

    def sample_beta_action(self) -> np.ndarray:
        """
        从 Beta(α,β) 中按形状参数采样，
        原始 u ∈ [0,1]，再线性映射到 action ∈ [-1,1]。
        返回 shape = (num_drones, DEFAULT_ACTION_DIM)
        """
        u = np.random.beta(self._beta_alpha,
                           self._beta_beta,
                           size=(self.NUM_DRONES, DEFAULT_ACTION_DIM))
        # 映射到 [-1,1]
        return (2.0 * u - 1.0).astype(np.float32)

    def _check_collision(self) -> bool:
        """
        若无人机与地面或任何障碍物接触 / 距离阈值内，则返回 True
        """
        drone_id = self._drone_id  # = self.DRONE_IDS[0]
        state = self._getDroneStateVector(0)
        ray_dists = self._cast_static_rays(state[0:3])  # 已包含可视化
        horiz_dists = ray_dists[self._horizontal_idx]  # 取俯仰=0° 的 N_H 根
        min_d = horiz_dists.min()
        # 始终打印最小距离
        if self.DEBUG:
            interval = max(1, self.MAX_STEPS // 10)
            if self.step_counter % interval == 0:
                print(f"[DEBUG] horizontal ray min distance = {min_d:.3f} m")
        if min_d < RAY_COLLISION_THRESH:
            if self.DEBUG:
                print(f"[COLLISION] horizontal ray min={min_d:.3f}m  <  {RAY_COLLISION_THRESH}")
            return True

        return False

    def _draw_rays(self, pos: np.ndarray, dists: np.ndarray):
        """在 GUI 里把本帧所有射线画出来."""
        if not (self.GUI and VIS_RAY_DEBUG):
            return

        # 1) 清除上一帧
        for rid in self._ray_vis_ids:
            p.removeUserDebugItem(rid, physicsClientId=self.CLIENT)
        self._ray_vis_ids.clear()

        # 2) 画新线
        dirs = self._ray_directions_body
        for i in range(self.N_H * self.N_V):
            end = pos + dirs[i] * dists[i]
            hit = dists[i] < RAY_LEN - 1e-3  # 命中障碍
            near = dists[i] < RAY_COLLISION_THRESH  # 距离过近
            color = ([1, 0, 0] if near else  # 红=已触发
                     [1, 1, 0] if hit else  # 黄=命中但安全
                     [0, 1, 0])  # 绿=未命中
            rid = p.addUserDebugLine(
                lineFromXYZ=pos, lineToXYZ=end, lineColorRGB=color,
                lifeTime=0, physicsClientId=self.CLIENT)
            self._ray_vis_ids.append(rid)