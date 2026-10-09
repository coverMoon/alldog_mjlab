"""Wolf flat 任务的单一人工训练参数入口（与 black_config.py 同风格）。

修改后由 env_cfgs.py / rl_cfg.py 组装为 MjLab v1.6.0 原生配置；这里不替代
ManagerBasedRlEnvCfg，也不定义 term 顺序或 policy joint order。机器人固有参数、
default pose 与执行器见 robots/wolf/wolf_constants.py。

POLICY / DEPLOYMENT CONTRACT-SENSITIVE：control timestep/decimation、action
scale 与 actor observation scales。修改它们须建立新训练契约并重验导出和部署兼容性。
"""

from dataclasses import dataclass, field
from typing import Literal
import math

from alldog_mjlab.tasks.velocity.black.black_config import (
    CommandCurriculumParams,
)


@dataclass(frozen=True)
class EnvParams:
    # Wolf 训练默认 4096 env，play 默认 1；CLI 显式指定时仍以 CLI 为准。
    train_num_envs: int = 4096
    play_num_envs: int = 1
    episode_length_s: float = 20.0


@dataclass(frozen=True)
class ControlParams:
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：policy dt = physics_dt * decimation = 0.02 s。
    physics_dt: float = 0.005
    decimation: int = 4
    # q_target = q_default + leg_action_scale * raw；dq_target = wheel_sign * 10 * raw。
    leg_action_scale: float = 0.20
    wheel_action_scale: float = 10.0

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
    wheel_vel_scale: float = 0.05
    last_action_scale: float = 1.0
    actor_corruption_enabled: bool = True


# =============================================================================
# Command
# =============================================================================


@dataclass(frozen=True)
class CommandParams:
    """Wolf command contract：初始 vx 范围 + native sampler + forward-speed 课程。

    生成器为 MjLab v1.6 `UniformVelocityCommand`。``lin_vel_x`` 是课程起点，训练中
    由性能驱动 curriculum 扩展到 ±4 m/s；``lin_vel_y`` / ``ang_vel_z`` 训练全程固定
    （范围小于 Black：轮足混合机器人在低速侧移 / 原地转下动力学不同，先收窄）。
    """

    resampling_time: tuple[float, float] = (10.0, 10.0)

    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-0.15, 0.15)
    ang_vel_z: tuple[float, float] = (-0.6, 0.6)

    standing_fraction: float = 0.1
    forward_fraction: float = 0.2
    world_fraction: float = 0.0
    init_velocity_prob: float = 0.0

    # 性能驱动 forward-speed 课程：评价状态机语义与 Black 完全一致
    # （CommandCurriculumParams，复用 Black 已验证的数据类），仅扩展 vx 并把
    # 目标范围改为 ±4.0 m/s；其余阈值 / 统计参数均为 Black 已验证默认值。
    command_curriculum: CommandCurriculumParams = field(
        default_factory=lambda: CommandCurriculumParams(max_abs_vx=4.0,
                                                        initial_lin_vel_x=(-1.0, 1.0))
    )


# =============================================================================
# Observation noise
# =============================================================================


@dataclass(frozen=True)
class NoiseParams:
    """Actor observation 各分量的 raw 值均匀噪声幅值（与 Black 相同档）。

    MjLab pipeline 为 compute → noise → clip → scale；进入 policy 的最终幅值 =
    raw noise × observation scale。wheel_vel 暂用与 joint_vel 相同的 raw 噪声档。
    """

    enabled: bool = True
    base_ang_vel: tuple[float, float] = (-0.3, 0.3)
    projected_gravity: tuple[float, float] = (-0.05, 0.05)
    joint_pos: tuple[float, float] = (-0.08, 0.08)
    joint_vel: tuple[float, float] = (-2.0, 2.0)
    wheel_vel: tuple[float, float] = (-2.0, 2.0)


# =============================================================================
# Reset
# =============================================================================


@dataclass(frozen=True)
class ResetParams:
    """Episode reset 的 pose / velocity 采样范围（小幅扰动，nominal dynamics）。

    root pose 不随机高度 / 姿态（default 站立 0.4432 m 直接落地）；root 六维速度
    小幅独立均匀采样；腿关节相对 default pose 小幅 offset（完全在 soft limit 内）；
    轮子位置 / 速度归零（default 即 0）。
    """

    root_pose: dict[str, tuple[float, float]] = field(default_factory=dict)

    root_velocity: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.3, 0.3),
            "roll": (-0.3, 0.3),
            "pitch": (-0.3, 0.3),
            "yaw": (-0.5, 0.5),
        }
    )

    leg_position: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hip": (0.0, 0.0),
            "thigh": (-0.3, 0.3),
            "calf": (-0.3, 0.3),
        }
    )

    # 轮子 reset：位置 / 速度均为 0（显式表达，不留隐式依赖）。
    wheel_position: tuple[float, float] = (0.0, 0.0)
    wheel_velocity: tuple[float, float] = (0.0, 0.0)


# =============================================================================
# Termination
# =============================================================================


@dataclass(frozen=True)
class TerminationParams:
    """终止阈值（第一版：仅 time_out + base_link 非法接触；无倾角 / 无 stuck）。"""

    illegal_contact_force: float = 1.0
    illegal_contact_history: int = 4
    illegal_contact_bodies: tuple[str, ...] = ("base_link",)


# =============================================================================
# Reward
# =============================================================================


@dataclass(frozen=True)
class RewardScales:
    tracking_linear: float = 1.0
    tracking_angular: float = 0.5
    lin_vel_z: float = -1.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.5
    base_height: float = -2.0
    leg_action_rate: float = -0.01
    wheel_action_rate: float = -0.002


@dataclass(frozen=True)
class RewardParams:
    """Wolf flat v1 reward baseline 的系数与非系数参数。

    tracking_sigma 是指数分母（HIMLoco 语义）；base_height_target 用 Wolf 完整模型
    实测（wheel r=0.08 / 站立稳态 0.3961 m），不沿用 Black 的 0.43。
    """

    tracking_sigma: float = 0.25
    base_height_target: float = 0.40
    scales: RewardScales = field(default_factory=RewardScales)


# =============================================================================
# Domain Randomization
# =============================================================================


@dataclass(frozen=True)
class CalfBacklashParams:
    """calf backlash（旧版 play 模式；在 action term 层实现，见 randomization.py）。"""

    enabled: bool = False
    width_range: tuple[float, float] = (0.005, 0.035)
    min_kp_scale: float = 0.12
    engage_start: float = 0.6
    leak: float = 0.02


@dataclass(frozen=True)
class WheelTargetBiasParams:
    """轮 velocity target 的 scaling / bias（per-episode 重采样）。"""

    vel_ref_scale_enabled: bool = False
    vel_ref_scale_range: tuple[float, float] = (0.9, 1.1)
    vel_ref_bias_enabled: bool = False
    vel_ref_bias_range: tuple[float, float] = (-0.3, 0.3)


@dataclass(frozen=True)
class WheelObsBiasParams:
    """轮速观测偏置（仅 actor 可观测通道；critic / reward / PD 用真实值）。"""

    enabled: bool = False
    range: tuple[float, float] = (-0.5, 0.5)


@dataclass(frozen=True)
class DomainRandomizationParams:
    """Wolf flat 的 domain randomization（旧 BlackW 数值为候选范围）。

    默认全部关闭（baseline 连续性）；每个功能有独立 ``*_enabled`` 开关，
    全关时与 nominal physics 行为逐位一致。play 模式强制全关。

    旧版中确认无效（LEGACY INERT / UNSUPPORTED）的项不在本参数集中，见
    MIGRATION.md §29 迁移对照表：

    - ``randomize_restitution``：MjLab v1.6 无可靠恢复系数机制 → 不迁移；
    - ``randomize_hip_damping``：旧 URDF dof damping=0，缩放无效 → 不迁移；
    - ``wheel_base_half_width_scale``：旧 learned 轮速模式未使用该参数 → 不迁移。

    delay 单位：policy step（env_cfgs 装配时换算为 physics step 注入原生
    actuator delay，换算 = ×decimation）。
    """

    # --- 刚体质量 / COM / 惯量（相对 nominal，per-episode 重采样）---
    base_mass_enabled: bool = False
    base_mass_range: tuple[float, float] = (-1.0, 2.0)
    base_com_enabled: bool = False
    base_com_offset_range: tuple[float, float] = (-0.05, 0.05)
    link_mass_enabled: bool = False
    link_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    link_inertia_enabled: bool = False
    link_inertia_scale_range: tuple[float, float] = (0.9, 1.1)
    wheel_mass_enabled: bool = False
    wheel_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    wheel_inertia_enabled: bool = False
    wheel_inertia_scale_range: tuple[float, float] = (0.8, 1.2)

    # --- 摩擦（MuJoCo pair 取 max，轮摩擦直接以绝对值写轮 geom）---
    ground_friction_enabled: bool = False
    ground_friction_range: tuple[float, float] = (0.25, 1.25)
    wheel_friction_enabled: bool = False
    wheel_friction_scale_range: tuple[float, float] = (0.4, 1.0)

    # --- 执行器（腿 PD / 电机强度 / calf backlash）---
    kp_enabled: bool = False
    kp_scale_range: tuple[float, float] = (0.9, 1.1)
    kd_enabled: bool = False
    kd_scale_range: tuple[float, float] = (0.9, 1.1)
    motor_strength_enabled: bool = False
    motor_strength_range: tuple[float, float] = (0.9, 1.1)
    hip_motor_strength_enabled: bool = False
    hip_motor_strength_range: tuple[float, float] = (0.8, 1.05)
    calf_backlash: CalfBacklashParams = field(default_factory=CalfBacklashParams)

    # --- 轮部执行器 / target / 观测 ---
    wheel_motor_enabled: bool = False
    wheel_motor_strength_range: tuple[float, float] = (0.8, 1.2)
    wheel_target: WheelTargetBiasParams = field(default_factory=WheelTargetBiasParams)
    wheel_obs_bias: WheelObsBiasParams = field(default_factory=WheelObsBiasParams)

    # --- 轮几何 ---
    wheel_radius_enabled: bool = False
    wheel_radius_scale_range: tuple[float, float] = (0.9, 1.1)

    # --- 控制延迟（单位 policy step；换算 physics step = ×decimation）---
    leg_delay_enabled: bool = False
    leg_max_delay_steps: int = 3
    wheel_delay_enabled: bool = False
    wheel_max_delay_steps: int = 4

    # --- 外部扰动 ---
    push_enabled: bool = False
    push_interval_s: tuple[float, float] = (15.0, 15.0)
    push_velocity_xy: tuple[float, float] = (-1.0, 1.0)
    disturbance_enabled: bool = False
    disturbance_force_range: tuple[float, float] = (-30.0, 30.0)
    disturbance_interval_policy_steps: int = 8

    # --- reset：initial joint position（multiplicative，旧版语义）---
    initial_joint_pos_enabled: bool = False
    initial_joint_pos_range: tuple[float, float] = (0.5, 1.5)

    def validate(self) -> None:
        """装配前的一致性检查（非法 / 不自洽时 fail-loud）。"""
        for name in (
            "base_mass_range", "base_com_offset_range", "link_mass_scale_range",
            "link_inertia_scale_range", "wheel_mass_scale_range",
            "wheel_inertia_scale_range", "ground_friction_range",
            "wheel_friction_scale_range", "kp_scale_range", "kd_scale_range",
            "motor_strength_range", "hip_motor_strength_range",
            "wheel_motor_strength_range", "wheel_radius_scale_range",
            "initial_joint_pos_range",
        ):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"dr.{name} 区间反序: {lo} > {hi}")
        for name in ("leg_max_delay_steps", "wheel_max_delay_steps",
                     "disturbance_interval_policy_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"dr.{name} 必须非负")


def replace_default_dr(**kwargs: object) -> DomainRandomizationParams:
    """以全关默认为基线构造 DR 参数（显式覆盖个别开关 / 范围）。"""
    import dataclasses as _dc

    return _dc.replace(DomainRandomizationParams(), **kwargs)  # type: ignore[arg-type]


# minimal 训练 profile：轮地摩擦 + 腿部 PD gains 小范围随机化，其余全关。
# 使用方式：把下方 WolfConfig() 构造改为 WolfConfig(dr=MINIMAL_DR)。
MINIMAL_DR = replace_default_dr(
    ground_friction_enabled=True, kp_enabled=True, kd_enabled=True
)


@dataclass(frozen=True)
class SimulationParams:
    # MJWarp per-world capacity（capacity tuning，不改 solver 数学）。
    # spawn 模板（qpos0，freejoint 起 z=0）下 Wolf 多个 collision 几何与 terrain
    # 平面相交：实测 124 contacts -> nefc 下限 496（= 124x4 pyramidal rows，
    # tile 16 对齐 -> 512）。运行时腿自接触大幅消失、只剩轮-地接触，nefc 预计远低
    # 于 512（诊断工具需保持 overflow NO；后续可按实测收窄）。nconmax 同样被
    # spawn 自接触抬高（>=124），Rough 阶段如有实测再调。
    nconmax: int = 128
    njmax: int = 512


@dataclass(frozen=True)
class PolicyParams:
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    # POLICY / DEPLOYMENT CONTRACT-SENSITIVE：actor 不做 running normalization。
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
class HimParams:
    """Wolf HIM estimator / latent 参数（与 Black 相同 official HIM 默认）。

    PPO 侧超参数继续复用 PolicyParams / AlgorithmParams。
    """

    latent_dim: int = 16
    encoder_hidden_dims: tuple[int, ...] = (128, 64)
    target_encoder_hidden_dims: tuple[int, ...] = (128, 64)
    num_prototypes: int = 32
    temperature: float = 3.0
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 10.0


@dataclass(frozen=True)
class StageRunnerParams:
    run_name: str
    load_run: str


@dataclass(frozen=True)
class RunnerParams:
    seed: int = 42
    experiment_name: str = "wolf_velocity"
    flat: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(run_name="wolf_flat", load_run=r".*_wolf_flat$")
    )
    flat_him: StageRunnerParams = field(
        default_factory=lambda: StageRunnerParams(
            run_name="wolf_flat_him", load_run=r".*_wolf_flat_him$"
        )
    )
    save_interval: int = 50
    num_steps_per_env: int = 64
    max_iterations: int = 10_000
    # resume 时 command curriculum state 的默认恢复策略（语义与 Black 一致：
    # auto = 同 stage full / 跨 stage range / 旧 checkpoint none；PPO→HIM warm start
    # 不经过本字段，固定 range）。
    command_curriculum_restore: Literal["auto", "none", "range", "full"] = "auto"


@dataclass(frozen=True)
class WolfConfig:
    env: EnvParams = field(default_factory=EnvParams)
    control: ControlParams = field(default_factory=ControlParams)
    command: CommandParams = field(default_factory=CommandParams)
    observation: ObservationParams = field(default_factory=ObservationParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    reset: ResetParams = field(default_factory=ResetParams)
    termination: TerminationParams = field(default_factory=TerminationParams)
    reward: RewardParams = field(default_factory=RewardParams)
    # DR：默认全关（baseline 连续性）；minimal 训练 profile 见 MINIMAL_DR。
    dr: DomainRandomizationParams = field(default_factory=DomainRandomizationParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    policy: PolicyParams = field(default_factory=PolicyParams)
    algorithm: AlgorithmParams = field(default_factory=AlgorithmParams)
    him: HimParams = field(default_factory=HimParams)
    runner: RunnerParams = field(default_factory=RunnerParams)


WOLF_CONFIG = WolfConfig()


if __name__ == "__main__":
    cc = WOLF_CONFIG.command.command_curriculum
    cc.validate()
    print("WOLF_CONFIG ok; curriculum max_abs_vx =", cc.max_abs_vx)
