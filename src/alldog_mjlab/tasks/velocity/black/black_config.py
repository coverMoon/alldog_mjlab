"""Black flat/rough 的单一人工训练参数入口。

修改后由 env_cfgs.py / terrain.py / rl_cfg.py 组装为 MjLab v1.6.0 原生配置；
这里不替代 ManagerBasedRlEnvCfg，也不定义 term 顺序或 policy joint order。
机器人固有参数、default pose 与 actuator 见 robots/black/。

POLICY / DEPLOYMENT CONTRACT-SENSITIVE：control timestep/decimation 与 actor
observation scales。修改它们须建立新训练契约并重验导出和部署兼容性。
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class EnvParams:
    # MjLab v1.6 velocity factory 默认 1，已验证训练命令另用 CLI 覆盖成 4096。
    # CLI 覆盖仍优先；若要从此文件控制训练数量，请移除 CLI 的 --env.scene.num-envs。
    num_envs: int = 1
    episode_length_s: float = 20.0


@dataclass(frozen=True)
class ControlParams:
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：policy dt 由两者相乘，不单独存储。
    physics_dt: float = 0.005
    decimation: int = 4

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation


@dataclass(frozen=True)
class ObservationParams:
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：仅数值 scale；term 顺序在 env_cfgs.py。
    command_scale: tuple[float, float, float] = (2.0, 2.0, 0.25)
    base_ang_vel_scale: float = 0.25
    projected_gravity_scale: float = 1.0
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.05
    last_action_scale: float = 1.0
    actor_corruption_enabled: bool = True


# =============================================================================
# Command
# =============================================================================


@dataclass(frozen=True)
class CommandParams:
    """Black flat v1 的最终 command contract（训练全程固定，无 curriculum）。

    生成器为 MjLab v1.6 `UniformVelocityCommand`：body-frame 速度指令，按
    `resampling_time` 重采样。行/角速度单位为 m/s 与 rad/s。
    """

    resampling_time: tuple[float, float] = (10.0, 10.0)

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-math.pi, math.pi)

    # native sampler 比例：standing env 指令置 0；forward-only env 取 vx >= 0.3
    # 且 vy = ang_vel_z = 0；world-frame env 与 reset 初速度随机均未启用。
    standing_fraction: float = 0.1
    forward_fraction: float = 0.2
    world_fraction: float = 0.0
    init_velocity_prob: float = 0.0


# =============================================================================
# Observation noise
# =============================================================================


@dataclass(frozen=True)
class NoiseParams:
    """Actor observation 各分量的 raw 值均匀噪声幅值。

    MjLab pipeline 为 compute → noise → clip → scale，因此这里写加在 raw 值上的
    噪声：进入 policy 的最终幅值 = raw noise × observation scale。
    """

    enabled: bool = True
    base_ang_vel: tuple[float, float] = (-0.3, 0.3)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    joint_pos: tuple[float, float] = (-0.08, 0.08)
    joint_vel: tuple[float, float] = (-2.0, 2.0)


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


# =============================================================================
# Reward
# =============================================================================


@dataclass(frozen=True)
class RewardScales:
    tracking_linear: float = 1.0
    tracking_angular: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.2
    base_height: float = -1.0
    dof_acc: float = -2.5e-7
    joint_power: float = -2e-5
    action_rate: float = -0.01
    smoothness: float = -0.01


@dataclass(frozen=True)
class RewardParams:
    """Black flat v1 reward baseline 的系数与非系数参数。

    tracking_sigma 是指数分母，不是 sigma²；base_height_target 是期望 root
    高度，与 reset 的 0.45 m 职责不同。权重集中在 scales。
    """

    tracking_sigma: float = 0.25
    base_height_target: float = 0.43
    scales: RewardScales = field(default_factory=RewardScales)


# =============================================================================
# Domain randomization
# =============================================================================


@dataclass(frozen=True)
class DomainRandomizationParams:
    """Black flat v1 的 domain randomization（train 生效，play 全部移除）。

    startup（每 env 采样一次并在 episode 内保持）：friction、payload_mass、
    com_offset、encoder_bias；
    reset（每次 episode reset 重新采样）：pd_gains；
    interval（训练中周期性施加 xy 速度增量）：push_interval / push_velocity。
    """

    friction_enabled: bool = True
    payload_enabled: bool = True
    com_enabled: bool = True
    pd_gain_enabled: bool = True
    encoder_bias_enabled: bool = True
    push_enabled: bool = True

    # Contact
    friction: tuple[float, float] = (0.2, 1.25)

    # Rigid body（trunk payload 与 nominal COM 的 offset）
    payload_mass: tuple[float, float] = (-1.0, 2.0)
    com_offset: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {
            0: (-0.05, 0.05),
            1: (-0.05, 0.05),
            2: (-0.05, 0.05),
        }
    )

    # Actuator（相对 nominal Kp=40 / Kd=1.2 的缩放）
    kp_scale: tuple[float, float] = (0.9, 1.1)
    kd_scale: tuple[float, float] = (0.9, 1.1)

    # Sensor（固定 encoder calibration bias，不是 observation noise）
    encoder_bias: tuple[float, float] = (-0.015, 0.015)

    # Disturbance（间隔 [s] 与 root xy 速度增量 [m/s]）
    push_interval: tuple[float, float] = (16.0, 16.0)
    push_velocity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
    )


# =============================================================================
# Terrain
# =============================================================================


@dataclass(frozen=True)
class TerrainParams:
    """Black rough v1 的 terrain generator 数值。

    patch 尺寸与高度场分辨率与 legacy Black 一致（4 m 半径 / 0.1 m / 0.005 m）。
    MjLab v1.6 curriculum 的 row difficulty 按 ``row / (num_rows - 1)`` 在
    ``difficulty_range`` 内插值，因此取 ``(0.0, 0.9)`` + ``num_rows = 10`` 以恢复
    legacy 的 ``row / num_rows`` 语义（0.0, 0.1, ..., 0.9，不含 1.0）。

    proportions 是 curriculum 模式下每类 terrain 的 env 分配权重（不是列数）。
    单项可以为 0；总和必须大于 0。
    """

    # Patch 与网格
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 10
    difficulty_range: tuple[float, float] = (0.0, 0.9)
    # MjLab border 为 z = 0 的 flat apron（与 legacy heightfield border_size=25 m
    # 语义不同），取 MjLab native rough preset 的 20.0 m。
    border_width: float = 20.0

    # 高度场分辨率（legacy 同值）
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    platform_width: float = 3.0

    # 各 terrain 的 env 分配权重
    proportions: dict[str, float] = field(
        default_factory=lambda: {
            "flat": 0.20,
            "smooth_slope_up": 0.15,
            "smooth_slope_down": 0.15,
            "rough_slope": 0.30,
            "discrete_obstacles": 0.20,
        }
    )

    # smooth slope：max slope = 0.7 x 0.9 = 0.63
    slope_range: tuple[float, float] = (0.0, 0.7)

    # rough slope 噪声：amplitude = rough_noise_base + rough_noise_gain x difficulty
    rough_noise_base: float = 0.015
    rough_noise_gain: float = 0.1
    rough_noise_step: float = 0.005
    rough_noise_downsample: float = 0.2
    rough_base_thickness_ratio: float = 1.0

    # discrete obstacles：height = 0.06 + difficulty x 0.2
    obstacle_height_range: tuple[float, float] = (0.06, 0.26)
    obstacle_width_range: tuple[float, float] = (1.0, 2.0)
    obstacle_count: int = 20

    # 初始 terrain level 上限（inclusive，与 legacy max_init_terrain_level 同义）
    max_init_terrain_level: int = 5

    def validate(self) -> None:
        expected = {
            "flat",
            "smooth_slope_up",
            "smooth_slope_down",
            "rough_slope",
            "discrete_obstacles",
        }
        if set(self.proportions) != expected:
            raise ValueError(f"Black terrain proportions must have keys {sorted(expected)}")
        if any(
            not math.isfinite(value) or value < 0
            for value in self.proportions.values()
        ):
            raise ValueError("Black terrain proportions must be finite and nonnegative")
        if sum(self.proportions.values()) <= 0:
            raise ValueError("Black terrain proportions must have positive total weight")


@dataclass(frozen=True)
class SimulationParams:
    # MJWarp GPU contact-capacity workaround；不是 legacy Black 行为。flat 保持 native 35。
    rough_nconmax: int = 128


@dataclass(frozen=True)
class PolicyParams:
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：actor 保持不做 running normalization。
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = True
    initial_std: float = 1.0
    std_type: str = "scalar"


@dataclass(frozen=True)
class AlgorithmParams:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class RunnerParams:
    # MjLab v1.6 runner 默认 seed=42；由官方 CLI 覆盖时以 CLI 为准。
    seed: int = 42
    experiment_name: str = "black_velocity"
    save_interval: int = 50
    num_steps_per_env: int = 24
    max_iterations: int = 10_000


@dataclass(frozen=True)
class BlackConfig:
    env: EnvParams = field(default_factory=EnvParams)
    control: ControlParams = field(default_factory=ControlParams)
    command: CommandParams = field(default_factory=CommandParams)
    observation: ObservationParams = field(default_factory=ObservationParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    reset: ResetParams = field(default_factory=ResetParams)
    termination: TerminationParams = field(default_factory=TerminationParams)
    reward: RewardParams = field(default_factory=RewardParams)
    domain_rand: DomainRandomizationParams = field(default_factory=DomainRandomizationParams)
    terrain: TerrainParams = field(default_factory=TerrainParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    policy: PolicyParams = field(default_factory=PolicyParams)
    algorithm: AlgorithmParams = field(default_factory=AlgorithmParams)
    runner: RunnerParams = field(default_factory=RunnerParams)


BLACK_CONFIG = BlackConfig()
