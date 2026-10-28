"""Wolf robot assets and control contract."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# MJCF and assets
_HERE = Path(__file__).parent

WOLF_XML: Path = _HERE / "xmls" / "wolf.xml"

WOLF_ASSETS_DIR: Path = _HERE / "xmls" / "assets"

assert WOLF_XML.exists(), f"Wolf MJCF not found: {WOLF_XML}"


def _mesh_files() -> tuple[str, ...]:
    import xml.etree.ElementTree as ET

    root = ET.parse(WOLF_XML).getroot()
    return tuple((mesh.get("file") for mesh in root.iter("mesh") if mesh.get("file")))


def _mesh_assets_complete() -> bool:
    files = _mesh_files()
    return bool(files) and all(((WOLF_ASSETS_DIR / f).is_file() for f in files))


def get_spec() -> mujoco.MjSpec:
    missing = [f for f in _mesh_files() if not (WOLF_ASSETS_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Wolf mesh assets missing: {missing}（meshdir={WOLF_ASSETS_DIR}）"
        )
    return mujoco.MjSpec.from_file(str(WOLF_XML))


# Joint / policy order
WOLF_LEG_ORDER = ("FL", "FR", "RL", "RR")

WOLF_JOINTS_PER_LEG = ("hip", "thigh", "calf", "foot")

WOLF_POLICY_JOINT_NAMES = tuple(
    (f"{leg}_{joint}" for leg in WOLF_LEG_ORDER for joint in WOLF_JOINTS_PER_LEG)
)

WOLF_LEG_JOINT_NAMES = {
    leg: (f"{leg}_hip", f"{leg}_thigh", f"{leg}_calf", f"{leg}_foot")
    for leg in WOLF_LEG_ORDER
}

WOLF_WHEEL_JOINT_NAMES = tuple((f"{leg}_foot" for leg in WOLF_LEG_ORDER))

WOLF_WHEEL_FORWARD_SIGN = {"FL": 1, "FR": -1, "RL": 1, "RR": -1}

WOLF_IMU_BODY_NAME = "imu_Link"

WOLF_IMU_SITE_NAME = "imu_site"

WOLF_IMU_ANG_VEL_SENSOR = "imu_ang_vel"

WOLF_IMU_UPVECTOR_SENSOR = "imu_upvector"

WOLF_IMU_BODY_OFFSET = (0.0, 0.0, 0.05975)


# Actuator configuration
WOLF_LEG_STIFFNESS = {"hip": 90.0, "thigh": 80.0, "calf": 80.0}

WOLF_LEG_DAMPING = {"hip": 3.3, "thigh": 3.0, "calf": 3.0}

WOLF_LEG_EFFORT_LIMIT = 60.0

WOLF_WHEEL_STIFFNESS = 0.0

WOLF_WHEEL_DAMPING = 2.0

WOLF_WHEEL_EFFORT_LIMIT = 17.0


def _actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
    joint_type = joint_name.rsplit("_", 1)[-1]
    if joint_type in WOLF_LEG_STIFFNESS:
        return IdealPdActuatorCfg(
            target_names_expr=(joint_name,),
            stiffness=WOLF_LEG_STIFFNESS[joint_type],
            damping=WOLF_LEG_DAMPING[joint_type],
            effort_limit=WOLF_LEG_EFFORT_LIMIT,
        )
    if joint_name in WOLF_WHEEL_JOINT_NAMES:
        return IdealPdActuatorCfg(
            target_names_expr=(joint_name,),
            stiffness=WOLF_WHEEL_STIFFNESS,
            damping=WOLF_WHEEL_DAMPING,
            effort_limit=WOLF_WHEEL_EFFORT_LIMIT,
        )
    raise KeyError(f"unknown Wolf joint: {joint_name}")


WOLF_ACTUATOR_CFGS = tuple((_actuator_cfg(name) for name in WOLF_POLICY_JOINT_NAMES))

WOLF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=WOLF_ACTUATOR_CFGS, soft_joint_pos_limit_factor=0.9
)


# Initial state
WOLF_DEFAULT_ROOT_Z = 0.4432

WOLF_DEFAULT_JOINT_POS = {
    "FL": (0.0, 0.82, 1.43, 0.0),
    "FR": (0.0, -0.82, -1.43, 0.0),
    "RL": (0.0, -0.82, -1.43, 0.0),
    "RR": (0.0, 0.82, 1.43, 0.0),
}

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, WOLF_DEFAULT_ROOT_Z),
    joint_pos={
        joint: value
        for leg in WOLF_LEG_ORDER
        for joint, value in zip(WOLF_LEG_JOINT_NAMES[leg], WOLF_DEFAULT_JOINT_POS[leg])
    },
    joint_vel={".*": 0.0},
)


# Collision configuration
# Collision geometry and physical parameters are loaded unchanged from the MJCF.
WOLF_WHEEL_COLLISION_GEOM_NAMES = tuple(
    (f"{leg}_wheel_collision" for leg in WOLF_LEG_ORDER)
)

WOLF_WHEEL_BODY_NAMES = tuple((f"{leg}_Link4" for leg in WOLF_LEG_ORDER))

WOLF_WHEEL_RADIUS = 0.08

WOLF_WHEEL_HALF_WIDTH = 0.0225


# Final robot configuration
def get_wolf_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=deepcopy(INIT_STATE),
        articulation=deepcopy(WOLF_ARTICULATION),
        sort_actuators=True,
    )


if __name__ == "__main__":
    from mjlab.entity.entity import Entity

    robot = Entity(get_wolf_robot_cfg())
    model = robot.spec.compile()
    print("Wolf Entity compiled successfully.")
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"nbody: {model.nbody}")
    print("\nJoints（MJCF natural order）:")
    for i in range(model.njnt):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    print("\nActuators（编译后 ctrl 顺序）:")
    for i in range(model.nu):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
