"""Black velocity task 的人工调参入口。

只放训练者预期会查看 / 调整的 numeric / range 参数，按 task 生命周期分成小组：

    params.command.*
    params.observation_noise.*
    params.reset.*
    params.termination.*
    params.reward.*

policy / task 的 interface contract（term 顺序、selector、observation scale、
sensor 身份、MjLab wiring）不属于调参，留在 env_cfgs.py。

本文件不是第二套 runtime config：MjLab ``ManagerBasedRlEnvCfg`` 仍是唯一 runtime
config，这里只是它的数值来源。
"""

from dataclasses import dataclass, field
import math

# =============================================================================
# Command
# =============================================================================


@dataclass(frozen=True)
class CommandParams:
    """Velocity command 数值范围（m/s、rad/s 与 s）。"""

    resampling_time: tuple[float, float] = (10.0, 10.0)

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-math.pi, math.pi)


command = CommandParams()


# =============================================================================
# Observation noise
# =============================================================================


@dataclass(frozen=True)
class ObservationNoiseParams:
    """Actor observation 各分量的 raw 值均匀噪声幅值。

    MjLab pipeline 为 compute → noise → clip → scale，因此这里写加在 raw 值上的
    噪声：进入 policy 的最终幅值 = raw noise × observation scale。
    """

    base_ang_vel: tuple[float, float] = (-0.3, 0.3)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    joint_pos: tuple[float, float] = (-0.08, 0.08)
    joint_vel: tuple[float, float] = (-2.0, 2.0)


observation_noise = ObservationNoiseParams()


# =============================================================================
# Reset
# =============================================================================


@dataclass(frozen=True)
class ResetParams:
    """Episode reset 的 pose / velocity 采样范围。

    root_pose 为空 dict 表示不随机（default pose + zero offset，root 高度取 default
    0.45 m）；root_velocity 的 key 为 MjLab v1.6.0 的 SE(3) 轴名。
    joint_position 是以 default joint pose 为均值的对称 offset，三组 support 均完全
    落在 soft joint limits 内（不依赖 clamp）；joint velocity 不随机。
    """

    root_pose: dict[str, tuple[float, float]] = field(default_factory=dict)

    root_velocity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }
    )

    joint_position: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hip": (0.0, 0.0),
            "thigh": (-0.4007, 0.4007),
            "calf": (-0.5945, 0.5945),
        }
    )

    joint_velocity: tuple[float, float] = (0.0, 0.0)


reset = ResetParams()


# =============================================================================
# Termination
# =============================================================================


@dataclass(frozen=True)
class TerminationParams:
    """终止阈值。

    illegal_contact_force [N]：trunk / thigh 与 terrain 接触力超过它即终止；
    illegal_contact_history 取一个 control step 内的 physics substep 数。
    stuck_* [s] / [m/s] / [m/s] / [s]：有效 planar command 下沿指令方向无 progress
    的连续时长超过 stuck_timeout 即终止，stuck_grace 内不计时。
    """

    illegal_contact_force: float = 1.0
    illegal_contact_history: int = 4

    stuck_timeout: float = 4.0
    stuck_velocity_threshold: float = 0.05
    stuck_command_threshold: float = 0.2
    stuck_grace: float = 1.0


termination = TerminationParams()


# =============================================================================
# Reward
# =============================================================================


@dataclass(frozen=True)
class RewardParams:
    """Black flat v1 reward baseline 的系数与非系数参数。

    除 tracking_sigma（指数分母，不是 sigma²）与 base_height_target（期望 root
    高度，与 reset 的 0.45 m 职责不同）外，其余字段都是 reward coefficient。
    """

    # Tracking
    tracking_sigma: float = 0.25
    tracking_linear: float = 1.0
    tracking_angular: float = 0.5

    # Base stability
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.2

    # Height
    base_height_target: float = 0.43
    base_height: float = -1.0

    # Regularization（dof_acc 为 control-step 关节速度有限差分的平方）
    dof_acc: float = -2.5e-7
    joint_power: float = -2e-5
    action_rate: float = -0.01
    smoothness: float = -0.01


reward = RewardParams()
