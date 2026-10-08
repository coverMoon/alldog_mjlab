"""从 MjLab / RSL-RL checkpoint 导出 actor 的 actor-only TorchScript（task-aware）。

同时支持普通 PPO 与 HIM 两条 policy（按 checkpoint 加载后的 actor 类型分支）；
输入 / 输出维度来自当前 task 的 env 与 policy 对象（不硬编码 Black 的 45/12），
并与 policy 网络结构交叉校验：

```text
Black PPO deployment contract（§19.3）：   input [1, 45]  / output [1, 12]
Black HIM deployment contract（§20.1）：   input [1, 270] / output [1, 12]
Wolf PPO deployment contract（§28）：      input [1, 53]  / output [1, 16]
Wolf HIM deployment contract（§28）：      input [1, 318] / output [1, 16]
```

实现不重建网络、不手工解析 checkpoint：

- checkpoint 加载走 MjLab runner 的正式 ``load()``（``load_cfg={"actor": True}``），
  保留 ``MjlabOnPolicyRunner.load()`` 的 checkpoint 格式兼容逻辑；
- 导出走 RSL-RL 5.4.2 原生 ``runner.export_policy_to_jit()``（PPO 用 MLPModel.as_jit，
  HIM 用 HIMPolicy.as_jit 的脚本化包装，两者都由 ``export_policy_to_jit`` 内部触发）；
- observation / action 维度来自当前 task 与 env / HIMSpec，不来自 checkpoint，
  并在导出前显式校验；history 长度 / canonical 维度按 HIMSpec 运行时读取（不硬编码）。

actor 无 observation normalization（§5.3），因此这里不做任何额外 normalization / scaling。

用法：

```bash
uv run export --task-id black-flat
uv run export --task-id black-flat-him
uv run export --task-id black-rough --load-run 2026-09-18_19-37-17 \
    --checkpoint model_498.pt --output-dir ~/models/black --device cpu
```

默认用 MjLab v1.6 ``get_checkpoint_path()`` 从
``logs/rsl_rl/<experiment_name>`` 按 task runner 的 ``load_run`` 选择最新同 stage
run（Black flat / rough 分别匹配 ``.*_flat$`` / ``.*_rough$``，HIM 为
``.*_flat_him$`` / ``.*_rough_him$``），再选择最新匹配 checkpoint，输出到
``<run>/exported/policy.pt``。显式 ``--load-run`` 可访问 legacy run。
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path

from alldog_mjlab.algorithms.him.policy import HIMPolicy
from alldog_mjlab.algorithms.him.spec import canonical_history

# 数值等价容差。
ATOL = 1e-6
RTOL = 1e-5
# 导出环境只用于构建 actor 与产生 probe，单 env 足够。
NUM_ENVS = 1
# probe 生成用的固定 seed（probe 本身仍由固定公式给出，不依赖随机数）。
PROBE_SEED = 42


def resolve_export_paths(
    task_id: str,
    load_run: str | None = None,
    checkpoint_pattern: str = "model_.*.pt",
    output_dir: Path | None = None,
) -> tuple[str, Path, Path]:
    """按 MjLab v1.6 规则选择 checkpoint，并确定固定文件名的输出路径。"""
    try:
        agent_cfg = load_rl_cfg(task_id)
    except KeyError as exc:
        raise ValueError(f"unknown task id: {task_id}") from exc
    experiment_name = agent_cfg.experiment_name
    log_root = Path("logs/rsl_rl") / experiment_name
    checkpoint = get_checkpoint_path(
        log_root,
        run_dir=agent_cfg.load_run if load_run is None else load_run,
        checkpoint=checkpoint_pattern,
    )
    export_dir = (
        checkpoint.parent / "exported" if output_dir is None else output_dir.expanduser()
    )
    return experiment_name, checkpoint, export_dir / "policy.pt"


def resolve_env_layout(wrapped: RslRlVecEnvWrapper) -> tuple[int, int, int]:
    """从当前 task 的 env 读取 (frame_dim, history_length, action_dim)。

    actor group 的 ``group_obs_dim`` 是 task 唯一权威：单帧任务为 ``(D,)``，
    HIM history 任务为 ``(H, D)``；action 维度来自 ActionManager。不硬编码
    Black 的 45/12/270。
    """
    manager = wrapped.unwrapped
    actor_dim = manager.observation_manager.group_obs_dim["actor"]
    if isinstance(actor_dim, tuple) and len(actor_dim) == 2:
        history_length, frame_dim = int(actor_dim[0]), int(actor_dim[1])
    else:
        # 单帧 group 的 group_obs_dim 为 (D,)（或 D）。
        frame_dim = int(actor_dim[0]) if isinstance(actor_dim, tuple) else int(actor_dim)
        history_length = 1
    action_dim = int(manager.action_manager.total_action_dim)
    return frame_dim, history_length, action_dim


def resolve_actor_layout(
    policy: torch.nn.Module,
    env_frame_dim: int,
    env_history_length: int,
) -> tuple[int, int]:
    """从 policy 对象确定 (frame_dim, history_length) 并与 env 交叉校验。

    PPO（MLPModel）为单帧（history_length = 1），第一层权重输入维度必须等于
    env 的 frame_dim；HIM（HIMPolicy）用 ``spec.single_frame_dim`` /
    ``spec.history_length`` 并与 env 维度交叉校验。task contract 判定不靠
    shape 猜测、也不靠魔数。
    """
    if isinstance(policy, HIMPolicy):
        spec = policy.spec
        frame_dim = int(spec.single_frame_dim)
        history_length = int(spec.history_length)
        if frame_dim != env_frame_dim:
            raise ValueError(
                f"HIM spec single_frame_dim = {frame_dim} != env frame dim {env_frame_dim}"
            )
        if history_length != env_history_length:
            raise ValueError(
                f"HIM spec history_length = {history_length} != env history {env_history_length}"
            )
    else:
        frame_dim = env_frame_dim
        history_length = 1
        input_dim = _policy_input_dim(policy)
        if input_dim != frame_dim:
            raise ValueError(
                f"PPO actor 第一层输入维度 {input_dim} != env frame dim {frame_dim}；"
                "checkpoint 与 task 不匹配"
            )
    return frame_dim, history_length


def _policy_input_dim(policy: torch.nn.Module) -> int:
    """读取 MLP actor 的第一层输入维度（不含 obs normalizer 时也一样成立）。"""
    for module in policy.modules():
        if isinstance(module, torch.nn.Linear):
            first_weight = module.weight
            if first_weight.ndim == 2:
                return int(first_weight.shape[1])
    raise ValueError("policy 中找不到 Linear 层，无法校验输入维度")


def check_env_contract(
    wrapped: RslRlVecEnvWrapper,
    frame_dim: int,
    history_length: int,
    action_dim: int,
) -> list[str]:
    """校验 actor observation / action 维度来自当前 task 的 env（不来自 checkpoint）。"""
    actor_obs = wrapped.get_observations()["actor"]
    expected_shape = (
        (NUM_ENVS, frame_dim) if history_length == 1 else (NUM_ENVS, history_length, frame_dim)
    )
    if tuple(actor_obs.shape) != expected_shape:
        raise ValueError(
            f"actor observation shape = {tuple(actor_obs.shape)}, expected {expected_shape}"
        )
    manager = wrapped.unwrapped.action_manager
    if int(manager.total_action_dim) != action_dim:
        raise ValueError(
            f"action dim = {manager.total_action_dim} != derived {action_dim}"
        )
    return list(manager.active_terms)


def collect_probes(
    wrapped: RslRlVecEnvWrapper,
    device: str,
    frame_dim: int,
    history_length: int,
    action_dim: int,
) -> tuple[list[tuple[str, torch.Tensor, torch.Tensor]], TensorDict]:
    """三类 deterministic probe：play 环境真实一帧 / 全零 / 固定 linspace。

    真实一帧取固定 action 走一步后的 actor observation，使 joint_pos / joint_vel /
    last_action 分量非平凡。返回 (label, policy 输入 probe, TorchScript 输入) 三元组：
    PPO 两者同为 [1, F]；HIM probe 为 [1, H, F] history，TorchScript 输入为
    canonical flatten [1, H×F]（newest → oldest）。

    linspace probe 的每帧乘以不同缩放（0.90 → 1.00），保证各帧数值不同，
    可以暴露 frame 顺序错误。
    """
    probe_action = torch.linspace(-1.0, 1.0, action_dim, device=device).repeat(NUM_ENVS, 1)
    obs = wrapped.step(probe_action)[0]
    real_history = obs["actor"].detach().clone()
    zeros_history = torch.zeros(
        (NUM_ENVS, frame_dim) if history_length == 1 else (NUM_ENVS, history_length, frame_dim),
        dtype=torch.float32,
        device=device,
    )
    frame_linspace = torch.linspace(-1.0, 1.0, frame_dim, dtype=torch.float32, device=device)
    frames = [frame_linspace * (0.90 + 0.02 * i) for i in range(history_length)]
    linspace_history = (
        frame_linspace.reshape(NUM_ENVS, frame_dim)
        if history_length == 1
        else torch.stack(frames, dim=0).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
    )
    probes = [
        ("play_env_frame", real_history, real_history),
        ("zeros", zeros_history, zeros_history),
        ("linspace", linspace_history, linspace_history),
    ]
    # HIM：TorchScript 输入是 canonical flatten（newest → oldest frame-major）。
    probes = [
        (label, probe, canonical_history(probe) if history_length > 1 else probe)
        for label, probe, _ in probes
    ]
    return probes, obs


def deterministic_output(
    policy: torch.nn.Module, obs: TensorDict, probe: torch.Tensor
) -> torch.Tensor:
    """RSL-RL actor 的 deterministic forward（把 actor slice 换成 probe）。"""
    values = {key: value for key, value in obs.items()}
    values["actor"] = probe
    with torch.inference_mode():
        return policy(TensorDict(values, batch_size=[NUM_ENVS])).detach().clone()


def check_probe_shapes(
    label: str, jit_input: torch.Tensor, output: torch.Tensor, action_dim: int
) -> None:
    """断言 TorchScript 输出满足部署 contract：[1, action_dim] 且为 float32。"""
    if tuple(output.shape) != (NUM_ENVS, action_dim):
        raise ValueError(f"[{label}] output shape = {tuple(output.shape)}")
    if jit_input.dtype != torch.float32 or output.dtype != torch.float32:
        raise ValueError(f"[{label}] dtype = {jit_input.dtype} / {output.dtype}")


def load_torchscript(path: Path, device: str) -> torch.jit.ScriptModule:
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    return module


def verify_export(
    output: Path,
    device: str,
    probes: list[tuple[str, torch.Tensor, torch.Tensor]],
    reference: dict[str, torch.Tensor],
    input_dim: int,
    action_dim: int,
) -> dict:
    """重新 torch.jit.load 导出的 .pt，与 checkpoint actor 逐 probe 比较数值等价。"""
    module = load_torchscript(output, device)
    rows = []
    for label, probe, jit_input in probes:
        if tuple(jit_input.shape) != (NUM_ENVS, input_dim):
            raise ValueError(
                f"[{label}] input shape = {tuple(jit_input.shape)}, expected {(NUM_ENVS, input_dim)}"
            )
        with torch.inference_mode():
            exported = module(jit_input).detach().clone()
        check_probe_shapes(label, jit_input, exported, action_dim)
        expected = reference[label]
        max_abs_diff = float((expected - exported).abs().max().item())
        allclose = bool(torch.allclose(expected, exported, atol=ATOL, rtol=RTOL))
        rows.append(
            {
                "probe": label,
                "input_shape": list(jit_input.shape),
                "output_shape": list(exported.shape),
                "dtype": str(exported.dtype).replace("torch.", ""),
                "max_abs_diff": max_abs_diff,
                "allclose": allclose,
            }
        )
        print(
            f"[{label}] input={list(jit_input.shape)} output={list(exported.shape)} "
            f"dtype={rows[-1]['dtype']} max_abs_diff={max_abs_diff:.3e} allclose={allclose}"
        )
        if not allclose:
            raise AssertionError(
                f"[{label}] numerical equivalence failed: max_abs_diff = {max_abs_diff:.6e} "
                f"> atol {ATOL:.1e} / rtol {RTOL:.1e}"
            )

    # 独立加载验证：.pt 不依赖 runner 对象即可加载并推理（默认 CPU）。
    standalone = torch.jit.load(str(output))
    standalone.eval()
    zeros = torch.zeros(NUM_ENVS, input_dim, dtype=torch.float32)
    with torch.inference_mode():
        standalone_out = standalone(zeros).detach().clone()
    if tuple(standalone_out.shape) != (NUM_ENVS, action_dim):
        raise ValueError(f"standalone output shape = {tuple(standalone_out.shape)}")
    if not bool(torch.isfinite(standalone_out).all()):
        raise AssertionError("standalone TorchScript output 含非有限值")
    print(
        f"[standalone] reloaded without runner: input={[NUM_ENVS, input_dim]} "
        f"output={list(standalone_out.shape)} finite=True"
    )
    return {"probes": rows, "standalone_reload": True, "atol": ATOL, "rtol": RTOL}


def run_export(task_id: str, checkpoint: Path, output: Path, device: str) -> dict:
    """构造 play 环境 → 加载 checkpoint actor → 导出 TorchScript → 数值等价验证。"""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    agent_cfg = load_rl_cfg(task_id)
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = PROBE_SEED
    env_cfg.scene.num_envs = NUM_ENVS
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device=device)

        # PPO（MLPModel，单帧）与 HIM（HIMPolicy，history [1,H,F] → canonical
        # [1, H×F]）共用同一验证骨架；维度全部来自当前 task 的 env + policy
        # 网络结构交叉校验（Black 导出仍应产生 45/12，Wolf 产生 53/16）。
        env_frame_dim, env_history_length, action_dim = resolve_env_layout(wrapped)
        frame_dim, history_length = resolve_actor_layout(
            policy, env_frame_dim, env_history_length
        )
        input_dim = frame_dim * history_length

        action_terms = check_env_contract(wrapped, frame_dim, history_length, action_dim)
        probes, obs = collect_probes(
            wrapped, device, frame_dim, history_length, action_dim
        )
        reference = {
            label: deterministic_output(policy, obs, probe)
            for label, probe, _ in probes
        }

        runner.export_policy_to_jit(str(output.parent), output.name)
        if not output.is_file():
            raise RuntimeError(f"TorchScript export 未产生文件: {output}")

        report = verify_export(
            output, device, probes, reference, input_dim, action_dim
        )
        report.update(
            {
                "task_id": task_id,
                "checkpoint": str(checkpoint),
                "output": str(output),
                "device": device,
                "num_envs": NUM_ENVS,
                "actor_frame_dim": frame_dim,
                "actor_history_length": history_length,
                "input_dim": input_dim,
                "action_dim": action_dim,
                "action_terms": action_terms,
                "obs_normalization": bool(agent_cfg.actor.obs_normalization),
            }
        )
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出并验证 MjLab checkpoint 中的 Black actor（PPO / HIM）")
    parser.add_argument("--task-id", required=True, help="已注册的 MjLab task ID")
    parser.add_argument("--load-run", help="run 目录名或正则；默认使用 task 的 stage 匹配规则")
    parser.add_argument(
        "--checkpoint",
        default="model_.*.pt",
        help="checkpoint 文件名或正则；默认 model_.*.pt",
    )
    parser.add_argument("--output-dir", type=Path, help="导出目录；默认 <run>/exported")
    parser.add_argument("--device", default="cpu", help="导出设备；默认 cpu")
    args = parser.parse_args()
    experiment, checkpoint, output = resolve_export_paths(
        args.task_id, args.load_run, args.checkpoint, args.output_dir
    )
    print(
        f"[export]\n"
        f"task:       {args.task_id}\n"
        f"experiment: {experiment}\n"
        f"run:        {checkpoint.parent}\n"
        f"checkpoint: {checkpoint}\n"
        f"output:     {output}\n"
        f"device:     {args.device}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    report = run_export(args.task_id, checkpoint, output, args.device)
    print(json.dumps(report, indent=2))
    print(f"PASS: exported actor-only TorchScript -> {output}")


if __name__ == "__main__":
    main()
