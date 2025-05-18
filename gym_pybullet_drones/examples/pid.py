import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# 默认无人机型号
DEFAULT_DRONES = DroneModel("cf2x")
# 默认无人机数量
DEFAULT_NUM_DRONES = 3
# 默认物理引擎模式
DEFAULT_PHYSICS = Physics("pyb")
# 是否启用图形界面
DEFAULT_GUI = True
# 是否录制视觉视频
DEFAULT_RECORD_VISION = False
# 是否绘图
DEFAULT_PLOT = True
# 是否显示用户调试线
DEFAULT_USER_DEBUG_GUI = True
# 是否添加障碍物
DEFAULT_OBSTACLES = True
# 仿真频率（Hz）
DEFAULT_SIMULATION_FREQ_HZ = 240
# 控制频率（Hz）
DEFAULT_CONTROL_FREQ_HZ = 48
# 仿真时长（秒）
DEFAULT_DURATION_SEC = 12
# 结果输出文件夹
DEFAULT_OUTPUT_FOLDER = 'results'
# 是否在 Colab 环境
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    # 初始化飞行高度、增量和半径
    H = 0.1
    H_STEP = 0.05
    R = 0.3
    # 生成初始位置阵列（螺旋上升）
    INIT_XYZS = np.array([
        [R*np.cos((i/6)*2*np.pi+np.pi/2),
         R*np.sin((i/6)*2*np.pi+np.pi/2)-R,
         H + i*H_STEP]
        for i in range(num_drones)
    ])
    # 生成初始姿态（偏航间隔分布）
    INIT_RPYS = np.array([
        [0, 0, i * (np.pi/2)/num_drones]
        for i in range(num_drones)
    ])

    # 生成圆形轨迹 waypoint
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = (
            R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2) + INIT_XYZS[0, 0],
            R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2) - R + INIT_XYZS[0, 1],
            0
        )
    wp_counters = np.array([int((i*NUM_WP/6) % NUM_WP) for i in range(num_drones)])

    # 创建飞行仿真环境
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )

    # 获取 PyBullet 客户端 ID
    PYB_CLIENT = env.getPyBulletClient()

    # 初始化日志记录器
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )

    # 初始化控制器列表（DSLPIDControl）
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    # 仿真主循环
    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(int(duration_sec * env.CTRL_FREQ)):
        # 执行一步仿真并获取观测
        obs, reward, terminated, truncated, info = env.step(action)

        # 计算当前 waypoint 的控制指令
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=np.hstack([
                    TARGET_POS[wp_counters[j], 0:2],
                    INIT_XYZS[j, 2]
                ]),
                target_rpy=INIT_RPYS[j, :]
            )

        # 更新 waypoint 计数器
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        # 记录本步数据
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([
                    TARGET_POS[wp_counters[j], 0:2],
                    INIT_XYZS[j, 2],
                    INIT_RPYS[j, :],
                    np.zeros(6)
                ])
            )

        # 渲染并同步
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    # 关闭环境，保存并绘制结果
    env.close()
    logger.save()
    logger.save_as_csv("pid")
    if plot:
        logger.plot()


if __name__ == "__main__":
    # 解析命令行参数并运行
    parser = argparse.ArgumentParser(
        description='使用 CtrlAviary 和 DSLPIDControl 的螺旋飞行示例'
    )
    parser.add_argument(
        '--drone', default=DEFAULT_DRONES, type=DroneModel,
        help='无人机型号 (默认: CF2X)', choices=DroneModel
    )
    parser.add_argument(
        '--num_drones', default=DEFAULT_NUM_DRONES, type=int,
        help='无人机数量 (默认: 3)'
    )
    parser.add_argument(
        '--physics', default=DEFAULT_PHYSICS, type=Physics,
        help='物理引擎 (默认: PYB)', choices=Physics
    )
    parser.add_argument(
        '--gui', default=DEFAULT_GUI, type=str2bool,
        help='是否启用 GUI (默认: True)'
    )
    parser.add_argument(
        '--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
        help='是否录制视频 (默认: False)'
    )
    parser.add_argument(
        '--plot', default=DEFAULT_PLOT, type=str2bool,
        help='是否绘图 (默认: True)'
    )
    parser.add_argument(
        '--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
        help='是否显示调试线 (默认: False)'
    )
    parser.add_argument(
        '--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
        help='是否添加障碍物 (默认: True)'
    )
    parser.add_argument(
        '--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
        help='仿真频率 (Hz)，默认 240'
    )
    parser.add_argument(
        '--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
        help='控制频率 (Hz)，默认 48'
    )
    parser.add_argument(
        '--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
        help='仿真时长 (秒)，默认 12'
    )
    parser.add_argument(
        '--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
        help='结果输出文件夹 (默认: "results")'
    )
    parser.add_argument(
        '--colab', default=DEFAULT_COLAB, type=bool,
        help='是否在 Colab 环境运行 (默认: False)'
    )
    ARGS = parser.parse_args()
    run(**vars(ARGS))