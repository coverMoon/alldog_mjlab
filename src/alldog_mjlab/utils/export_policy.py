"""从 MjLab / RSL-RL checkpoint 导出 Black PPO actor 的 actor-only TorchScript。

导出的 deployment contract（见 .ai/MIGRATION.md §19.3）：

```text
input  : float32 [1, 45]   actor 单帧 observation
output : float32 [1, 12]   policy action
```

实现不重建网络、不手工解析 checkpoint：

- checkpoint 加载走 MjLab runner 的正式 ``load()``（``load_cfg={"actor": True}``），
  保留 ``MjlabOnPolicyRunner.load()`` 的 checkpoint 格式兼容逻辑；
- 导出走 RSL-RL 5.4.2 原生 ``runner.export_policy_to_jit()``，内部使用 deterministic
  actor 的 ``as_jit()``；
- observation / action 维度来自当前 task 与 env，不来自 checkpoint，并在导出前显式校验。

actor 无 observation normalization（§5.3），因此这里不做任何额外 normalization / scaling。

用法：

```bash
uv run python -m alldog_mjlab.utils.export_policy \
    --task-id black-rough \
    --checkpoint logs/rsl_rl/black_velocity/<run>/model_498.pt \
    --output /tmp/policy.pt \
    --device cpu
```
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

# Black PPO actor contract（冻结值，见 .ai/MIGRATION.md §5 / §19.1）。
ACTOR_OBS_DIM = 45
ACTION_DIM = 12
# 数值等价容差。
ATOL = 1e-6
RTOL = 1e-5
# 导出环境只用于构建 actor 与产生 probe，单 env 足够。
NUM_ENVS = 1
# probe 生成用的固定 seed（probe 本身仍由固定公式给出，不依赖随机数）。
PROBE_SEED = 42


def check_env_contract(wrapped: RslRlVecEnvWrapper) -> list[str]:
    """校验 actor observation / action 维度来自当前 task 的 env（不来自 checkpoint）。"""
    obs_dim = int(wrapped.get_observations()["actor"].shape[-1])
    if obs_dim != ACTOR_OBS_DIM:
        raise ValueError(f"actor observation dim = {obs_dim} != {ACTOR_OBS_DIM}")
    manager = wrapped.unwrapped.action_manager
    action_dim = int(manager.total_action_dim)
    if action_dim != ACTION_DIM:
        raise ValueError(f"action dim = {action_dim} != {ACTION_DIM}")
    return list(manager.active_terms)


def collect_probes(
    wrapped: RslRlVecEnvWrapper, device: str
) -> tuple[list[tuple[str, torch.Tensor]], TensorDict]:
    """三类 deterministic probe：play 环境真实一帧 / 全零 / 固定 linspace。

    真实一帧取固定 action 走一步后的 actor observation，使 joint_pos / joint_vel /
    last_action 分量非平凡。probe 构造为 float32 [1, 45]。
    """
    probe_action = torch.linspace(-1.0, 1.0, ACTION_DIM, device=device).repeat(NUM_ENVS, 1)
    obs = wrapped.step(probe_action)[0]
    real_frame = obs["actor"].detach().clone()
    probes = [
        ("play_env_frame", real_frame),
        ("zeros", torch.zeros(NUM_ENVS, ACTOR_OBS_DIM, dtype=torch.float32, device=device)),
        (
            "linspace",
            torch.linspace(-1.0, 1.0, ACTOR_OBS_DIM, dtype=torch.float32, device=device).reshape(
                NUM_ENVS, ACTOR_OBS_DIM
            ),
        ),
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


def check_probe_shapes(label: str, probe: torch.Tensor, output: torch.Tensor) -> None:
    """断言 probe / 输出满足部署 contract：[1, 45] -> [1, 12] 且为 float32。"""
    if tuple(probe.shape) != (NUM_ENVS, ACTOR_OBS_DIM):
        raise ValueError(f"[{label}] input shape = {tuple(probe.shape)}")
    if tuple(output.shape) != (NUM_ENVS, ACTION_DIM):
        raise ValueError(f"[{label}] output shape = {tuple(output.shape)}")
    if probe.dtype != torch.float32 or output.dtype != torch.float32:
        raise ValueError(f"[{label}] dtype = {probe.dtype} / {output.dtype}")


def load_torchscript(path: Path, device: str) -> torch.jit.ScriptModule:
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    return module


def verify_export(
    output: Path,
    device: str,
    probes: list[tuple[str, torch.Tensor]],
    reference: dict[str, torch.Tensor],
) -> dict:
    """重新 torch.jit.load 导出的 .pt，与 checkpoint actor 逐 probe 比较数值等价。"""
    module = load_torchscript(output, device)
    rows = []
    for label, probe in probes:
        with torch.inference_mode():
            exported = module(probe).detach().clone()
        check_probe_shapes(label, probe, exported)
        expected = reference[label]
        max_abs_diff = float((expected - exported).abs().max().item())
        allclose = bool(torch.allclose(expected, exported, atol=ATOL, rtol=RTOL))
        rows.append(
            {
                "probe": label,
                "input_shape": list(probe.shape),
                "output_shape": list(exported.shape),
                "dtype": str(exported.dtype).replace("torch.", ""),
                "max_abs_diff": max_abs_diff,
                "allclose": allclose,
            }
        )
        print(
            f"[{label}] input={list(probe.shape)} output={list(exported.shape)} "
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
    with torch.inference_mode():
        standalone_out = standalone(torch.zeros(NUM_ENVS, ACTOR_OBS_DIM)).detach().clone()
    check_probe_shapes("standalone_zeros", torch.zeros(NUM_ENVS, ACTOR_OBS_DIM), standalone_out)
    if not bool(torch.isfinite(standalone_out).all()):
        raise AssertionError("standalone TorchScript output 含非有限值")
    print(
        f"[standalone] reloaded without runner: input={[NUM_ENVS, ACTOR_OBS_DIM]} "
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

        action_terms = check_env_contract(wrapped)
        probes, obs = collect_probes(wrapped, device)
        reference = {
            label: deterministic_output(policy, obs, probe) for label, probe in probes
        }

        runner.export_policy_to_jit(str(output.parent), output.name)
        if not output.is_file():
            raise RuntimeError(f"TorchScript export 未产生文件: {output}")

        report = verify_export(output, device, probes, reference)
        report.update(
            {
                "task_id": task_id,
                "checkpoint": str(checkpoint),
                "output": str(output),
                "device": device,
                "num_envs": NUM_ENVS,
                "actor_obs_dim": ACTOR_OBS_DIM,
                "action_dim": ACTION_DIM,
                "action_terms": action_terms,
                "obs_normalization": bool(agent_cfg.actor.obs_normalization),
            }
        )
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="导出的 .pt 文件路径")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_export(args.task_id, args.checkpoint, args.output, args.device)
    print(json.dumps(report, indent=2))
    print(f"PASS: exported actor-only TorchScript -> {args.output}")


if __name__ == "__main__":
    main()
