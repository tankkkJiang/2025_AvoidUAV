# ./control/CTBRControl.py

import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources
import socket
import struct

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel


class CTBRControl(object):
    """
    控制器基类

    实现 __init__(), reset(), computeControlFromState() 方法，
    具体的 computeControl() 需由子类实现。
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8
                 ):
        """
        通用控制器构造函数

        参数
        ----------
        drone_model : DroneModel
            待控制的无人机型号，对应 assets 文件夹下的 URDF 文件。
        g : float, optional
            重力加速度（m/s^2），默认为 9.8。
        """
        # 设置通用常量
        self.DRONE_MODEL = drone_model  # 无人机型号
        # 重力作用力 = 质量 * 重力加速度
        self.GRAVITY = g * self._getURDFParameter('m')
        # RPM 转推力系数
        self.KF = self._getURDFParameter('kf')
        # RPM 转扭矩系数
        self.KM = self._getURDFParameter('km')

        self.reset()

    ################################################################################

    def reset(self):
        """
        重置控制器状态

        将内部计数器置零。
        """
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3)
                                ):
        """
        从观测状态计算控制量接口

        接收 BaseAviary.step() 返回的 obs 中的 state，
        提取位置、四元数、速度等信息后调用 computeControl()。
        """
        return self.computeControl(
            control_timestep=control_timestep,
            cur_pos=state[0:3],
            # PyBullet 输出四元数需调整为 (w, x, y, z)
            cur_quat=np.array([state[6], state[3], state[4], state[5]]),
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates
        )

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
        抽象方法：计算单架无人机的控制输出

        需由子类实现或覆盖此方法。
        """
        # 检查输入维度
        assert (cur_pos.shape == (3,)), f"cur_pos {cur_pos.shape}"
        assert (cur_quat.shape == (4,)), f"cur_quat {cur_quat.shape}"
        assert (cur_vel.shape == (3,)), f"cur_vel {cur_vel.shape}"
        assert (cur_ang_vel.shape == (3,)), f"cur_ang_vel {cur_ang_vel.shape}"
        assert (target_pos.shape == (3,)), f"target_pos {target_pos.shape}"
        assert (target_rpy.shape == (3,)), f"target_rpy {target_rpy.shape}"
        assert (target_vel.shape == (3,)), f"target_vel {target_vel.shape}"
        assert (target_rpy_rates.shape == (3,)), f"target_rpy_rates {target_rpy_rates.shape}"

        # 重力向量
        G = np.array([0.0, 0.0, -9.8])
        # 位置环和速度环 PID 参数
        K_P = np.array([3., 3., 8.])
        K_D = np.array([2.5, 2.5, 5.])
        K_RATES = np.array([5., 5., 1.])
        # 位置误差和速度误差
        P = target_pos - cur_pos
        D = target_vel - cur_vel
        # 计算期望加速度
        tar_acc = K_P * P + K_D * D - G
        # 将加速度投影到机体 z 轴，得到归一化推力
        norm_thrust = np.dot(tar_acc, rotate_vector([0.0, 0.0, 1.0], cur_quat))
        # 计算期望机体姿态（四元数）
        z_body = normalized_vector(tar_acc)
        x_body = normalized_vector(np.cross(np.array([0.0, 1.0, 0.0]), z_body))
        y_body = normalized_vector(np.cross(z_body, x_body))
        tar_att = mat2quat(np.vstack([x_body, y_body, z_body]).T)
        # 计算角速度指令
        q_error = qmult(qconjugate(cur_quat), tar_att)
        body_rates = 2 * K_RATES * q_error[1:]
        # 如果标量部分为负，则翻转方向
        if q_error[0] < 0:
            body_rates = -body_rates

        return norm_thrust, *body_rates

    ################################################################################

    def setPIDCoefficients(self,
                           p_coeff_pos=None,
                           i_coeff_pos=None,
                           d_coeff_pos=None,
                           p_coeff_att=None,
                           i_coeff_att=None,
                           d_coeff_att=None
                           ):
        """
        设置 PID 控制器参数

        如果实例中不存在对应 PID 属性，则报错退出。
        """
        ATTR_LIST = [
            'P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR',
            'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR'
        ]
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] CTBRControl.setPIDCoefficients(): 未找到所有 PID 参数属性。")
            exit()
        else:
            self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
            self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
            self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
            self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
            self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
            self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    ################################################################################

    def _getURDFParameter(self,
                          parameter_name: str
                          ):
        """
        从 URDF 文件读取参数值

        解析 assets 目录下对应型号的 URDF，并返回指定参数。
        """
        # 构造 URDF 文件路径
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + URDF)
        URDF_TREE = etxml.parse(path).getroot()
        # 根据参数名提取并返回
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in [
            'arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh',
            'gnd_eff_coeff', 'prop_radius', 'drag_coeff_xy',
            'drag_coeff_z', 'dw_coeff_1', 'dw_coeff_2', 'dw_coeff_3'
        ]:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            offsets = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return offsets[2]