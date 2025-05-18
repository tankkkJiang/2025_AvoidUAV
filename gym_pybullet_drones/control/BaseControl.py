# ./control/BaseControl.py

import os
import numpy as np
import xml.etree.ElementTree as etxml
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel


class BaseControl(object):
    """
    控制器基类

    实现了 __init__(), reset(), computeControlFromState() 方法。
    具体的 computeControl() 方法需由子类实现。
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
            待控制的无人机型号，对应 `assets` 目录下的 URDF 文件。
        g : float, optional
            重力加速度（m/s^2），默认 9.8。
        """
        # 设置通用常量
        self.DRONE_MODEL = drone_model  # 无人机型号
        # 重力作用力 = 质量 * 重力加速度
        self.GRAVITY = g * self._getURDFParameter('m')
        # RPM 转推力的系数
        self.KF = self._getURDFParameter('kf')
        # RPM 转扭矩的系数
        self.KM = self._getURDFParameter('km')
        self.reset()

    ################################################################################

    def reset(self):
        """
        重置控制器状态

        将内部计数器清零。
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
        提取位置、速度等信息后调用 computeControl()。
        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
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

        需由子类实现。
        """
        raise NotImplementedError

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

        如果控制器未定义相应 PID 属性，则报错退出。
        """
        ATTR_LIST = ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR', 'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] BaseControl.setPIDCoefficients(): 未在实例中找到所有 PID 参数属性。")
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
        从 URDF 文件中读取参数值

        解析 `assets` 目录下对应型号的 URDF XML。

        参数
        ----------
        parameter_name : str
            要读取的参数名。

        返回
        -------
        float
            对应的参数数值。
        """
        # 构造 URDF 文件路径
        URDF = self.DRONE_MODEL.value + ".urdf"
        path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/' + URDF)
        URDF_TREE = etxml.parse(path).getroot()
        # 根据参数名查找并返回
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 'gnd_eff_coeff'
                                                                                     'prop_radius', 'drag_coeff_xy',
                                'drag_coeff_z', 'dw_coeff_1',
                                'dw_coeff_2', 'dw_coeff_3']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
