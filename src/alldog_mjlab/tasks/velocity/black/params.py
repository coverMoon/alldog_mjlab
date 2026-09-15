"""Black velocity task 的人工调参入口。

只放训练者预期会查看 / 调整的 numeric / range 参数。
policy 与 task 的 interface contract（term 顺序、selector、scale、sensor 身份、
MjLab wiring）留在 env_cfgs.py 或各自的实现模块里。

本文件不是第二套 runtime config：MjLab ``ManagerBasedRlEnvCfg`` 仍是唯一 runtime
config，这里只是它的参数来源。
"""

import math

# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

BLACK_COMMAND_RESAMPLING_TIME_RANGE = (10.0, 10.0)

BLACK_COMMAND_LIN_VEL_X_RANGE = (-1.0, 1.0)
BLACK_COMMAND_LIN_VEL_Y_RANGE = (-1.0, 1.0)
BLACK_COMMAND_ANG_VEL_Z_RANGE = (-math.pi, math.pi)

# ---------------------------------------------------------------------------
# Actor observation noise
# ---------------------------------------------------------------------------
# 各分量的 raw 值均匀噪声幅值。MjLab v1.6.0 pipeline 为
# compute → noise → clip → scale，因此这里写加在 raw 值上的噪声：
# 进入 policy 的最终幅值 = raw noise × observation scale。

BLACK_ACTOR_OBS_NOISE: dict[str, tuple[float, float]] = {
    "base_ang_vel": (-0.3, 0.3),
    "projected_gravity": (-0.05, 0.05),
    "joint_pos": (-0.08, 0.08),
    "joint_vel": (-2.0, 2.0),
}

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
# Root pose 不随机（x/y/z/roll/pitch/yaw 全部为 0 offset，root 高度取 default 0.45 m）；
# root 六维速度独立均匀采样。key 名称固定为 MjLab v1.6.0 的 SE(3) 轴名。

BLACK_ROOT_RESET_POSE_RANGE: dict[str, tuple[float, float]] = {}
BLACK_ROOT_RESET_VELOCITY_RANGE: dict[str, tuple[float, float]] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
}

# 以 default joint pose 为均值的对称 offset 采样，joint 速度不随机。
# 三个分组的 offset 范围均完全落在 MjLab soft joint limits 内（不依赖 clamp）：
#   hip   保持 default（offset 0）；
#   thigh ±0.4007 = default magnitude 的一半；
#   calf  ±0.5945，使左右 calf 的 support 都不超出 soft limits。

BLACK_JOINT_RESET_POSITION_RANGE: dict[str, tuple[float, float]] = {
    "hip": (0.0, 0.0),
    "thigh": (-0.4007, 0.4007),
    "calf": (-0.5945, 0.5945),
}
BLACK_JOINT_RESET_VELOCITY_RANGE: tuple[float, float] = (0.0, 0.0)

# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
# 摔倒终止：trunk 或任一 thigh 与 terrain 接触且接触力超过阈值 [N]。
# history_length 取一个 control step 内的 physics substep 数（0.005 × 4 = 0.02 s）。

BLACK_ILLEGAL_CONTACT_FORCE_THRESHOLD = 1.0
BLACK_ILLEGAL_CONTACT_HISTORY = 4

# Stuck 终止：planar command 有效（norm > command_threshold）且沿该指令方向的
# progress speed 持续低于 velocity_threshold 时累计计时，连续超过 timeout 即终止；
# grace 期内不计时。阈值单位分别为 m/s 与 s。

BLACK_STUCK_TIMEOUT_S = 4.0
BLACK_STUCK_VELOCITY_THRESHOLD = 0.05
BLACK_STUCK_COMMAND_THRESHOLD = 0.2
BLACK_STUCK_GRACE_S = 1.0

# ---------------------------------------------------------------------------
# Reward — tracking
# ---------------------------------------------------------------------------
# 指数速度跟踪。sigma 直接是 legacy 的 tracking_sigma（denominator 不是 sigma²）。

BLACK_TRACKING_SIGMA = 0.25
BLACK_TRACKING_LINEAR_WEIGHT = 2.0
BLACK_TRACKING_ANGULAR_WEIGHT = 1.5

# ---------------------------------------------------------------------------
# Reward — base motion stability
# ---------------------------------------------------------------------------
# 与 tracking 解耦的两个 penalty：lin_vel_z 罚 body-frame v_z²，
# body_ang_vel 罚 body-frame ω_x² + ω_y²。

BLACK_LIN_VEL_Z_WEIGHT = -2.0
BLACK_ANG_VEL_XY_WEIGHT = -0.05

# ---------------------------------------------------------------------------
# Reward — orientation
# ---------------------------------------------------------------------------
# 姿态 L1 惩罚的权重（raw = |g_x^b| + |g_y^b|）。
# legacy 的 terrain-adaptive pitch scaling 在当前 baseline 中关闭，故无自适应参数。

BLACK_ORIENTATION_WEIGHT = -0.8
