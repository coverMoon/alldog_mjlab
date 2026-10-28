"""Black domain randomization parameters, events and control perturbations."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainRandomizationParams:
    friction_enabled: bool = True
    payload_enabled: bool = True
    com_enabled: bool = True
    pd_gain_enabled: bool = True
    encoder_bias_enabled: bool = True
    push_enabled: bool = True
    friction: tuple[float, float] = (0.2, 1.25)
    payload_mass: tuple[float, float] = (-1.0, 2.0)
    com_offset: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
    )
    kp_scale: tuple[float, float] = (0.9, 1.1)
    kd_scale: tuple[float, float] = (0.9, 1.1)
    encoder_bias: tuple[float, float] = (-0.015, 0.015)
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
