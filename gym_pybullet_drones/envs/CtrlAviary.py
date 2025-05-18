import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class CtrlAviary(BaseAviary):
    """
    多无人机控制环境类，继承自 BaseAviary
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        """
        构造函数：初始化控制环境参数

        参数:
        drone_model : DroneModel
            无人机机型，对应 assets 文件夹下 URDF 定义
        num_drones : int
            无人机数量
        neighbourhood_radius : float
            邻域半径，用于计算邻接矩阵
        initial_xyzs : ndarray 或 None
            初始位置数组，形状 (NUM_DRONES, 3)
        initial_rpys : ndarray 或 None
            初始姿态角数组 (r,p,y)，形状 (NUM_DRONES, 3)
        physics : Physics
            物理引擎类型（PyBullet / 自定义）
        pyb_freq : int
            PyBullet 模拟步频，必须是 ctrl_freq 的整数倍
        ctrl_freq : int
            环境步频，即控制器调用频率
        gui : bool
            是否启用 PyBullet GUI
        record : bool
            是否录制仿真视频
        obstacles : bool
            是否在场景中添加障碍物
        user_debug_gui : bool
            是否绘制调试辅助线和 RPM 滑条
        output_folder : str
            日志和结果保存目录
        """
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            output_folder=output_folder
        )

    ################################################################################

    def _actionSpace(self):
        """
        定义动作空间：每架无人机 4 个电机的 RPM 指令
        返回:
        spaces.Box, 形状 (NUM_DRONES, 4)
        """
        # 最小/最大 RPM
        act_lower_bound = np.array([
            [0., 0., 0., 0.] for _ in range(self.NUM_DRONES)
        ])
        act_upper_bound = np.array([
            [self.MAX_RPM] * 4 for _ in range(self.NUM_DRONES)
        ])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32)

    ################################################################################

    def _observationSpace(self):
        """
        定义观测空间：每架无人机 20 维状态向量
        返回:
        spaces.Box, 形状 (NUM_DRONES, 20)
        """
        # 状态下界：位置(x,y无限制,z≥0)，四元数 [-1,1]，欧拉角 ±pi，速度无限制，角速度无限制，RPM 下界 0
        obs_lower_bound = np.array([
            [-np.inf, -np.inf, 0.,  # x,y,z
             -1., -1., -1., -1.,  # q1,q2,q3,q4
             -np.pi, -np.pi, -np.pi,  # roll,pitch,yaw
             -np.inf, -np.inf, -np.inf,  # vx,vy,vz
             -np.inf, -np.inf, -np.inf,  # wx,wy,wz
             0., 0., 0., 0.]  # PWM/RPM
            for _ in range(self.NUM_DRONES)
        ])
        # 状态上界：位置无限制，四元数上界 1，欧拉角 ±pi，速度无限制，角速度无限制，RPM 上界 MAX_RPM
        obs_upper_bound = np.array([
            [np.inf, np.inf, np.inf,
             1., 1., 1., 1.,
             np.pi, np.pi, np.pi,
             np.inf, np.inf, np.inf,
             np.inf, np.inf, np.inf,
             self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
            for _ in range(self.NUM_DRONES)
        ])
        return spaces.Box(low=obs_lower_bound,
                          high=obs_upper_bound,
                          dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """
        获取当前观测：调用 _getDroneStateVector
        返回:
        ndarray, 形状 (NUM_DRONES, 20)
        """
        return np.array([
            self._getDroneStateVector(i)
            for i in range(self.NUM_DRONES)
        ])

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """
        将输入的动作裁剪并格式化为电机可执行的 RPM 指令
        参数:
        action : ndarray, 任意实数 RPM 指令
        返回:
        ndarray, 裁剪后 (NUM_DRONES,4) 的 RPM 整数
        """
        return np.array([
            np.clip(action[i, :], 0, self.MAX_RPM)
            for i in range(self.NUM_DRONES)
        ])

    ################################################################################

    def _computeReward(self):
        """
        计算奖励，不适用于强化学习，返回固定占位值
        """
        return -1

    ################################################################################

    def _computeTerminated(self):
        """
        是否终止，不用于本环境，始终 False
        """
        return False

    ################################################################################

    def _computeTruncated(self):
        """
        是否截断，不用于本环境，始终 False
        """
        return False

    ################################################################################

    def _computeInfo(self):
        """
        返回附加信息，不用于本环境，始终 {'answer':42}
        """
        return {"answer": 42}  # 哲学深问的答案
