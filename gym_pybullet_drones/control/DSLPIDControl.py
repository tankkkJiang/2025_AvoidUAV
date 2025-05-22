# ./control/DSLPIDControl.py

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

DEFAULT_MAX_TILT_DEG = 45.0          # 最大倾角上限
DEFAULT_MAX_PWM      = 100000        # 最大 PWM 饱和
DEFAULT_MAX_SPEED_MPS= 25.0

class DSLPIDControl(BaseControl):
    """
    PID 控制类，适用于 Crazyflie 系列无人机
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 max_tilt_deg: float = DEFAULT_MAX_TILT_DEG,  # 最大倾角上限
                 max_pwm: int = DEFAULT_MAX_PWM,        # 最大 PWM 饱和
                 max_speed_mps: float = DEFAULT_MAX_SPEED_MPS
        ):
        """
        初始化 PID 控制器

        参数:
        drone_model : DroneModel
            无人机型号，对应 assets 文件夹中的 URDF
        g : float, optional
            重力加速度，单位 m/s^2
        """
        super().__init__(drone_model=drone_model, g=g)
        # 仅支持 CF2X 或 CF2P 机型
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] DSLPIDControl 仅支持 DroneModel.CF2X 或 CF2P")
            exit()
        # 位置环 PID 参数 （前向）
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])

        # 姿态环 PID 参数（扭矩）
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        # PWM 到 RPM 转换系数
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        # PWM 输出范围
        self.MAX_TILT_RAD = math.radians(max_tilt_deg)
        self.MIN_PWM = 0
        self.MAX_PWM = max_pwm
        self.max_speed_mps = max_speed_mps
        # 根据机型选择混合矩阵（动力分配），电机模型尽量不更改
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],
                [-.5, .5, 1],
                [.5, .5, -1],
                [.5, -.5, 1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],
                [+1, 0, 1],
                [0, 1, -1],
                [-1, 0, 1]
            ])
        # 重置内部状态
        self.reset()

    ################################################################################

    def reset(self):
        """
        重置控制器状态：位置和姿态的误差累积清零
        """
        super().reset()
        # 上一次的姿态角 rpy
        self.last_rpy = np.zeros(3)
        # 位置环误差
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        # 姿态环误差
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """
        计算一次 PID 控制：先位置控制，再姿态控制，返回每个电机 RPM 输出

        返回:
        rpm : ndarray
            4 个电机的转速（RPM）
        pos_e : ndarray
            当前位置误差 (x,y,z)
        yaw_error : float
            当前偏航角误差
        """
        self.control_counter += 1
        # 计算位置环得到推力和期望姿态角
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(
            control_timestep,
            cur_pos,
            cur_quat,
            cur_vel,
            target_pos,
            target_rpy,
            target_vel
        )
        # 根据推力和期望姿态角计算电机转速
        rpm = self._dslPIDAttitudeControl(
            control_timestep,
            thrust,
            cur_quat,
            computed_target_rpy,
            target_rpy_rates
        )
        # 计算当前偏航角误差
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        """
        位置环 PID 控制，输出推力和期望姿态角
        """
        # 当前旋转矩阵
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        # 位置误差、速度误差
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        # 累积分量，限幅防积分饱和
        self.integral_pos_e += pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        # PID 计算目标推力 (含重力补偿)
        target_thrust = (self.P_COEFF_FOR * pos_e
                         + self.I_COEFF_FOR * self.integral_pos_e
                         + self.D_COEFF_FOR * vel_e
                         + np.array([0, 0, self.GRAVITY]))
        # 将三轴推力投影到机身 z 轴方向并转为电机推力指令
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        # 计算期望机体坐标系轴向，为姿态控制做准备
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c)
        target_y_ax /= np.linalg.norm(target_y_ax)
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).T
        # 转换为欧拉角
        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', False)

        return thrust, target_euler, pos_e

    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               target_euler,
                               target_rpy_rates
                               ):
        """
        姿态环 PID 控制，输出每电机 PWM，再转为 RPM
        """
        # 当前旋转矩阵和欧拉角
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        # 目标四元数及旋转矩阵
        target_quat = Rotation.from_euler('XYZ', target_euler, False).as_quat()
        target_rotation = Rotation.from_quat(target_quat).as_matrix()
        # 计算旋转误差矩阵及误差向量
        rot_matrix_e = target_rotation.T @ cur_rotation - cur_rotation.T @ target_rotation
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        # 角速度误差
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy
        # 累积分量，限幅
        self.integral_rpy_e -= rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[:2] = np.clip(self.integral_rpy_e[:2], -1., 1.)
        # PID 计算目标扭矩
        target_torques = (- self.P_COEFF_TOR * rot_e
                          + self.D_COEFF_TOR * rpy_rates_e
                          + self.I_COEFF_TOR * self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        # PWM 混合
        pwm = thrust + self.MIXER_MATRIX.dot(target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        # 转为 RPM 输出
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """
        支持 1/2/3 维推力输入，转换为 4 通道 PWM
        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust) / (self.KF * (4 / DIM))) - self.PWM2RPM_CONST)
                      / self.PWM2RPM_SCALE,
                      self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4 // DIM)
        elif DIM == 2:
            return np.hstack([pwm, pwm[::-1]])
        else:
            print("[ERROR] _one23DInterface 支持维度为 1,2,4")
            exit()
