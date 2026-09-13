"""Black quadruped robot configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_HERE = Path(__file__).parent
BLACK_XML: Path = _HERE / "xmls" / "black.xml"

assert BLACK_XML.exists(), f"Black MJCF not found: {BLACK_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(BLACK_XML))


#
# Actuator
#

# Nominal control parameters from HIMLoco BlackCfg.
# Keep the RL thigh damping exception from the source configuration.
STIFFNESS = 40.0
DAMPING = 1.2
EFFORT_LIMIT = 20.0

# Position offsets in radians per unit action; independent of PD gains.
BLACK_ACTION_SCALE = 0.25

BLACK_FOOT_NAMES = ("FL", "FR", "RR", "RL")
BLACK_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in BLACK_FOOT_NAMES
    for joint in ("hip", "thigh", "calf")
)

BLACK_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=tuple(name for name in BLACK_JOINT_NAMES if name != "RL_thigh_joint"),
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=EFFORT_LIMIT,
)


BLACK_RL_THIGH_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=("RL_thigh_joint",),
    stiffness=STIFFNESS,
    damping=1.0,
    effort_limit=EFFORT_LIMIT,
)


BLACK_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(BLACK_ACTUATOR_CFG, BLACK_RL_THIGH_ACTUATOR_CFG),
    soft_joint_pos_limit_factor=0.9,
)


#
# Initial state
#

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.45),
    joint_pos={
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.8014,
        "FL_calf_joint": -1.527,

        "FR_hip_joint": 0.0,
        "FR_thigh_joint": -0.8014,
        "FR_calf_joint": 1.527,

        "RR_hip_joint": 0.0,
        "RR_thigh_joint": -0.8014,
        "RR_calf_joint": 1.527,

        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.8014,
        "RL_calf_joint": -1.527,
    },
    joint_vel={".*": 0.0},
)


#
# Final robot config
#

def get_black_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=INIT_STATE,
        articulation=BLACK_ARTICULATION,
        sort_actuators=True,
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    robot = Entity(get_black_robot_cfg())
    model = robot.spec.compile()

    print("Black Entity compiled successfully.")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"nbody: {model.nbody}")

    print("\nJoints:")
    for i in range(model.njnt):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

    print("\nActuators:")
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))