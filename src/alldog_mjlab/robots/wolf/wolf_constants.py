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


def _collision_only_spec_text() -> str:
    """从权威 wolf.xml 派生 collision-only MJCF 文本。

    规则（ElementTree 结构化处理，不做字符串正则替换）：
    - 删除全部 <mesh> asset 与 <geom type="mesh">（视觉 mesh geom 与其 asset）；
    - 保留 uninertial 标签、碰撞 geom（group=3 的 box / cylinder）、joint、
      材质定义与刚体结构；
    - 兼容 STL 缺失 / 损坏：无 mesh 引用后不再触发文件读取。

    imu body 无 joint 且无显式 inertial；collision-only 下该 body 为零质量焊死体，
    MuJoCo 允许（不产生 dof）。
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(WOLF_XML)
    root = tree.getroot()

    for asset in root.findall("./asset"):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    mesh_geoms = [geom for geom in parent_map if _tag_is_geom(geom) and geom.get("type") == "mesh"]
    for geom in mesh_geoms:
        parent_map[geom].remove(geom)
    ET.indent(tree)
    return ET.tostring(root, encoding="unicode")


def _tag_is_geom(element) -> bool:
    import xml.etree.ElementTree as ET

    # strip 命名空间后比对 raw tag
    return element.tag.rsplit("}", 1)[-1] == "geom"


def get_spec() -> mujoco.MjSpec:
    """返回 Wolf 的 MJCF spec；优先完整视觉模型，STL 缺失/损坏时退化 collision-only。

    加载模式（full / collision-only）明确打印，便于诊断动力学一致性来源。
    """
    if _mesh_assets_complete():
        try:
            spec = mujoco.MjSpec.from_file(str(WOLF_XML))
            spec.compile()  # STL 解码失败（如损坏 ASCII）在此暴露
            print("[INFO] Wolf spec mode: full visual model（STL 完整）")
            return spec
        except Exception as exc:  # noqa: BLE001  明确退化并记录原因
            reason = str(exc).splitlines()[0]
            print(
                f"[WARN] Wolf spec mode: collision-only（full visual 加载失败: {reason}）"
            )
        spec = mujoco.MjSpec.from_string(_collision_only_spec_text())
        return spec
    missing = [f for f in _mesh_files() if not (WOLF_ASSETS_DIR / f).is_file()]
    print(f"[INFO] Wolf spec mode: collision-only（mesh 缺失: {missing or 'n/a'}）")
    return mujoco.MjSpec.from_string(_collision_only_spec_text())


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
