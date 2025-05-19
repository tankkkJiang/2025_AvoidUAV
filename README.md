# AvoidUAV

## 1. 实现目标

我们的目标是在 PyBullet 中复现 NavRL 的无人机训练过程，主要是为了让无人机在动态障碍的环境下实现导航的功能。

我们实现了环境`NavRLAviary.py`，来自于是单机强化学习环境，继承自 BaseRLAviary，为 PPO、SAC 等算法准备好动作空间与观测空间，并实现了任务的奖励函数。

```bash
git clone https://github.com/tankkkJiang/2025_AvoidUAV.git
cd 2025_AvoidUAV

git pull origin main

pip install -e .
```

## 2. 代码架构
```bash
2025_AvoidUAV.
├── gym_pybullet_drones/       # 核心 Python 包
│   ├── __init__.py            # 包初始化
│   ├── assets/                # URDF、EEPROM 等模型与固件文件
│   │   ├── crazyflie.cf2x.urdf
│   │   ├── eeprom.bin
│   │   └── ...
│   ├── envs/                  # 各种飞控与RL环境（继承 BaseAviary / BaseRLAviary）
│   │   ├── BaseAviary.py
│   │   ├── BaseRLAviary.py
│   │   ├── NavRLAviary.py
│   │   ├── HoverAviary.py
│   │   └── ...  
│   ├── control/               # 控制器模块，将速度／姿态指令映射为电机推力
│   │   ├── BaseControl.py
│   │   ├── DSLPIDControl.py
│   │   └── ...
│   ├── utils/                 # 枚举类型、日志、WandB 回调等工具
│   │   ├── enums.py
│   │   ├── Logger.py
│   │   ├── wandb_callback.py
│   │   └── ...
│   └── examples/              # 示例脚本
│       ├── pid.py
│       ├── pid_velocity.py
│       ├── learn.py           # PPO 强化学习示例
│       ├── learn_navrl.py
│       ├── test_navrl_gui.py
│       └── ...
├── pyproject.toml             # 包管理与依赖配置
├── README.md                  # 项目说明
├── pypi_description.md        # PyPI 发布说明
└── LICENSE
```

## 3. 关键环境和代码
环境请参考：`NavRLAviary.py`；训练请参考：`learn_Nav.py`

先检查并去除原有虚拟显示：
```bash
ps aux | grep Xvfb
kill <PID>
```

使用下述指令启动 Xvfb + x11vnc   (在同一个终端或后台启动)
```bash
Xvfb :4 -screen 0 1280x720x24 & 
x11vnc -display :4 -rfbport 44445 -nopw &
export DISPLAY=:4
```

然后运行：
```bash
python3 learnNav.py
```

## 参考资料
https://github.com/Zhefan-Xu/NavRL?tab=readme-ov-file#V-Citation-and-Reference
https://github.com/utiasDSL/gym-pybullet-drones