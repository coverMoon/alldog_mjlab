"""Wolf wheel-legged robot configuration.

MJCF（``xmls/wolf.xml``）是模型权威来源；本文件定义 entity 级配置（初始状态、
逐关节 PD actuator）与显式 policy joint / wheel contract。

关节名保持 Wolf 原生命名（``FL_hip / FL_thigh / FL_calf / FL_foot``，其余腿同理），
不向 Black 的 ``*_joint`` 命名靠拢。

PD 参数均为首版仿真候选值，**不是已确认的实机电机规格**：
- 腿部 hip/thigh/calf：position PD，Kp=60、Kd=2.0、effort_limit=60 N·m；
- 轮部 foot：velocity PD，Kp=0、Kd=1.0、effort_limit=17 N·m（速度目标由后续
  task 的 velocity action 提供；轮部不做 torque action，也不用 XML velocity servo）。
"""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_HERE = Path(__file__).parent
WOLF_XML: Path = _HERE / "xmls" / "wolf.xml"
WOLF_ASSETS_DIR: Path = _HERE / "xmls" / "assets"

assert WOLF_XML.exists(), f"Wolf MJCF not found: {WOLF_XML}"


# ---------------------------------------------------------------------------
# 显式 joint / policy contract（不依赖 MJCF natural order、dict 插入顺序或 regex）
# ---------------------------------------------------------------------------

# Policy / action / deployment 腿顺序：FL → FR → RL → RR。
WOLF_LEG_ORDER = ("FL", "FR", "RL", "RR")
# 每腿 joint 顺序：hip → thigh → calf → foot（foot = 驱动轮）。
WOLF_JOINTS_PER_LEG = ("hip", "thigh", "calf", "foot")
# Policy joint 顺序（16 个）：腿间由 WOLF_LEG_ORDER、腿内由 WOLF_JOINTS_PER_LEG 决定。
WOLF_POLICY_JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in WOLF_LEG_ORDER
    for joint in WOLF_JOINTS_PER_LEG
)
# 每条腿的 joint 名（按 leg 键查询；无序依赖）。
WOLF_LEG_JOINT_NAMES = {
    leg: (
        f"{leg}_hip",
        f"{leg}_thigh",
        f"{leg}_calf",
        f"{leg}_foot",
    )
    for leg in WOLF_LEG_ORDER
}
# 轮关节（每腿第 4 个 joint，即 Link4 驱动轮）。
WOLF_WHEEL_JOINT_NAMES = tuple(f"{leg}_foot" for leg in WOLF_LEG_ORDER)
# 轮子的碰撞 geom 名（XML 中 Link4 的碰撞圆柱，仅命名不改几何）。
WOLF_WHEEL_COLLISION_GEOM_NAMES = tuple(
    f"{leg}_wheel_collision" for leg in WOLF_LEG_ORDER
)
# 轮子符号 contract：策略正轮速动作 = 该轮为机身前进方向（base +x）驱动。
# 验证依据（默认姿态 FK）：FL/RL 的轮轴 world 方向为 +y（a × ẑ = +x̂），
# FR/RR 的轮轴 world 方向为 -y（a × ẑ = -x̂），因此 MuJoCo joint target
# 为 policy 目标乘以该符号。
WOLF_WHEEL_FORWARD_SIGN = {
    "FL": 1,
    "FR": -1,
    "RL": 1,
    "RR": -1,
}

# 轮几何（XML 冻结值；测试用于 lowest-point / rolling 验证）。
WOLF_WHEEL_RADIUS = 0.1
WOLF_WHEEL_HALF_WIDTH = 0.0225

#
# IMU（imu_Link body 内原生 site + XML sensor；scene 中按 "<entity>/imu_*" 访问）
#

WOLF_IMU_BODY_NAME = "imu_Link"
WOLF_IMU_SITE_NAME = "imu_site"
# XML sensor 名（scene 访问时带实体名前缀，例如 "robot/imu_ang_vel"）。
WOLF_IMU_ANG_VEL_SENSOR = "imu_ang_vel"
WOLF_IMU_UPVECTOR_SENSOR = "imu_upvector"
# IMU 相对 base_link 的安装位置偏移（MJCF 冻结值；轴向当前与 base 平行，
# 实机安装方向待核对）。
WOLF_IMU_BODY_OFFSET = (0.0, 0.0, 0.05975)


# ---------------------------------------------------------------------------
# 显式 actuator / PD contract
# ---------------------------------------------------------------------------

# 腿部（hip/thigh/calf）position PD —— 首版仿真候选，非实机电机规格。
WOLF_LEG_STIFFNESS = 60.0
WOLF_LEG_DAMPING = 2.0
WOLF_LEG_EFFORT_LIMIT = 60.0
# 轮部（foot）velocity PD：Kp=0 → τ = Kd * (vel_target - dq)，clamp ±17。
WOLF_WHEEL_STIFFNESS = 0.0
WOLF_WHEEL_DAMPING = 1.0
WOLF_WHEEL_EFFORT_LIMIT = 17.0

_LEG_JOINTS = ("hip", "thigh", "calf")


def _actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
    if any(joint_name.endswith(f"_{joint}") for joint in _LEG_JOINTS):
        return IdealPdActuatorCfg(
            target_names_expr=(joint_name,),
            stiffness=WOLF_LEG_STIFFNESS,
            damping=WOLF_LEG_DAMPING,
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


# 逐关节 16 个 IdealPdActuatorCfg，覆盖 WOLF_POLICY_JOINT_NAMES。
# sort_actuators=True 时 MjLab 会按 model natural order 排列编译后的 actuator /
# ctrl 顺序，它与 policy action 顺序是两个概念（下一阶段的 action mapping 负责）。
WOLF_ACTUATOR_CFGS = tuple(_actuator_cfg(name) for name in WOLF_POLICY_JOINT_NAMES)

WOLF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=WOLF_ACTUATOR_CFGS,
    soft_joint_pos_limit_factor=0.9,
)


#
# Initial state（root z=0.45 为暂定值，待仿真确认；单位 quat）
#

WOLF_DEFAULT_ROOT_Z = 0.45

# 默认关节姿态（此前确认的 Wolf 值，单位 rad）。
# 注：thigh 与 calf 共用同一腿间符号映射（FL/RR 为正、FR/RL 为负）。
WOLF_DEFAULT_JOINT_POS = {
    "FL": (0.0, 0.82, 1.52, 0.0),
    "FR": (0.0, -0.82, -1.52, 0.0),
    "RL": (0.0, -0.82, -1.52, 0.0),
    "RR": (0.0, 0.82, 1.52, 0.0),
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


#
# Final robot config
#


def _mesh_files() -> tuple[str, ...]:
    """MJCF <asset> 里引用的全部 mesh 相对文件名（按声明顺序）。"""
    import xml.etree.ElementTree as ET

    root = ET.parse(WOLF_XML).getroot()
    return tuple(
        mesh.get("file") for mesh in root.iter("mesh") if mesh.get("file")
    )


def _mesh_assets_complete() -> bool:
    files = _mesh_files()
    return bool(files) and all((WOLF_ASSETS_DIR / f).is_file() for f in files)


def get_spec() -> mujoco.MjSpec:
    """返回 Wolf 的完整 MJCF spec（本地 STL 资产已完整，单模式加载）。

    STL 文件在本地完整存在但被 .gitignore 排除 Git 跟踪（已知且有意的资产管
    理方式）；不做双模式 / collision-only 回退。资源引用缺失时直接报错，
    不静默绕过。
    """
    missing = [f for f in _mesh_files() if not (WOLF_ASSETS_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Wolf mesh assets missing: {missing}（meshdir={WOLF_ASSETS_DIR}）"
        )
    return mujoco.MjSpec.from_file(str(WOLF_XML))


def get_wolf_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=get_spec,
        init_state=INIT_STATE,
        articulation=WOLF_ARTICULATION,
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
