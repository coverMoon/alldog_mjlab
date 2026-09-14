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

# 逐关节 PD 参数表，与旧 super-dog BlackCfg.control 的 dict 表达方式一致，
# 每个关节可独立调参，为后续差异化 PD 留出接口。
BLACK_STIFFNESS = {
    "FL_hip_joint": 40.0,
    "FL_thigh_joint": 40.0,
    "FL_calf_joint": 40.0,

    "FR_hip_joint": 40.0,
    "FR_thigh_joint": 40.0,
    "FR_calf_joint": 40.0,

    "RR_hip_joint": 40.0,
    "RR_thigh_joint": 40.0,
    "RR_calf_joint": 40.0,

    "RL_hip_joint": 40.0,
    "RL_thigh_joint": 40.0,
    "RL_calf_joint": 40.0,
}

BLACK_DAMPING = {
    "FL_hip_joint": 1.2,
    "FL_thigh_joint": 1.2,
    "FL_calf_joint": 1.2,

    "FR_hip_joint": 1.2,
    "FR_thigh_joint": 1.2,
    "FR_calf_joint": 1.2,

    "RL_hip_joint": 1.2,
    "RL_thigh_joint": 1.2,
    "RL_calf_joint": 1.2,

    "RR_hip_joint": 1.2,
    "RR_thigh_joint": 1.2,
    "RR_calf_joint": 1.2,
}

BLACK_EFFORT_LIMIT = {
    "FL_hip_joint": 20.0,
    "FL_thigh_joint": 20.0,
    "FL_calf_joint": 20.0,

    "FR_hip_joint": 20.0,
    "FR_thigh_joint": 20.0,
    "FR_calf_joint": 20.0,

    "RL_hip_joint": 20.0,
    "RL_thigh_joint": 20.0,
    "RL_calf_joint": 20.0,

    "RR_hip_joint": 20.0,
    "RR_thigh_joint": 20.0,
    "RR_calf_joint": 20.0,
}

# Position offsets in radians per unit action; independent of PD gains.
BLACK_ACTION_SCALE = 0.25

# Policy / action / deployment 权威腿顺序（对齐旧 super-dog Black 在 Isaac Gym
# 运行时打印的 self.dof_names：FL → FR → RL → RR）。
# 注意：这不是 MuJoCo MJCF 的 natural joint order（XML 中为 FL → FR → RR → RL），
# 两者是不同概念，不得混用。MJCF 顺序由 robot.joint_names 反映，保持不动。
BLACK_FOOT_NAMES = ("FL", "FR", "RL", "RR")
# Policy action / deployment 权威关节顺序：每条腿内 hip → thigh → calf，
# 腿间顺序由 BLACK_FOOT_NAMES 决定（FL → FR → RL → RR）。
BLACK_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in BLACK_FOOT_NAMES
    for joint in ("hip", "thigh", "calf")
)


def _pd_actuator_cfg(joint_name: str) -> IdealPdActuatorCfg:
    """按关节名从逐关节参数表生成单个关节的 PD actuator 配置。"""
    return IdealPdActuatorCfg(
        target_names_expr=(joint_name,),
        stiffness=BLACK_STIFFNESS[joint_name],
        damping=BLACK_DAMPING[joint_name],
        effort_limit=BLACK_EFFORT_LIMIT[joint_name],
    )


# 逐关节 12 个 PD actuator，覆盖 BLACK_JOINT_NAMES 的全部关节。
# 注意：sort_actuators=True 时 MjLab 会按 model natural order 排列编译后
# actuator 内部顺序，与 policy action 顺序无关；policy mapping 由
# env_cfgs 中的四个单腿 action term 负责。
BLACK_ACTUATOR_CFGS = tuple(_pd_actuator_cfg(name) for name in BLACK_JOINT_NAMES)

BLACK_ARTICULATION = EntityArticulationInfoCfg(
    actuators=BLACK_ACTUATOR_CFGS,
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