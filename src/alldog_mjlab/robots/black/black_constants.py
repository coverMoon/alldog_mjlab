"""Black robot assets and control contract."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# MJCF and assets
_HERE = Path(__file__).parent

BLACK_XML: Path = _HERE / "xmls" / "black.xml"

assert BLACK_XML.exists(), f"Black MJCF not found: {BLACK_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(BLACK_XML))


# Joint / policy order
BLACK_FOOT_NAMES = ("FL", "FR", "RL", "RR")

BLACK_JOINT_NAMES = tuple(
    (
        f"{leg}_{joint}_joint"
        for leg in BLACK_FOOT_NAMES
        for joint in ("hip", "thigh", "calf")
    )
)


# Actuator configuration
BLACK_STIFFNESS = {name: 40.0 for name in BLACK_JOINT_NAMES}

BLACK_DAMPING = {name: 1.2 for name in BLACK_JOINT_NAMES}

BLACK_EFFORT_LIMIT = {name: 20.0 for name in BLACK_JOINT_NAMES}


def _pd_actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
    return IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=BLACK_STIFFNESS[joint_name],
        damping=BLACK_DAMPING[joint_name],
        effort_limit=BLACK_EFFORT_LIMIT[joint_name],
    )


BLACK_ACTUATOR_CFGS = tuple((_pd_actuator_cfg(name) for name in BLACK_JOINT_NAMES))

BLACK_ARTICULATION = EntityArticulationInfoCfg(
    actuators=BLACK_ACTUATOR_CFGS, soft_joint_pos_limit_factor=0.9
)


# Initial state
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.45),
    joint_pos={
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.8014,
        "FL_calf_joint": -1.527,
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": -0.8014,
        "FR_calf_joint": 1.527,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.8014,
        "RL_calf_joint": -1.527,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": -0.8014,
        "RR_calf_joint": 1.527,
    },
    joint_vel={".*": 0.0},
)


# Collision configuration
# Collision geometry and physical parameters are loaded unchanged from the MJCF.


# Final robot configuration
def get_black_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=deepcopy(INIT_STATE),
        articulation=deepcopy(BLACK_ARTICULATION),
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
