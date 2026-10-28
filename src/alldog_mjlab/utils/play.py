"""Play a task; infer the latest checkpoint from its experiment when no explicit file is supplied."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from mjlab.tasks.registry import list_tasks, load_rl_cfg


def _argv_flag_value(prefix: str) -> str | None:
    """从 argv 找 ``--flag value`` / ``--flag=value`` 形式的值（找不到 None）。"""
    for i, arg in enumerate(sys.argv):
        if arg == prefix and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(f"{prefix}="):
            return arg.split("=", 1)[1]
    return None


def _latest_checkpoint_by_step(run_dir: Path) -> Path | None:
    """``model_*.pt`` 中 step 最大者（解析数字最大；无数字时按文件名兜底）。"""
    models = list(run_dir.glob("model_*.pt"))
    if not models:
        return None

    def step_of(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(models, key=lambda p: (step_of(p), p.name))


def resolve_latest_checkpoint(log_root: str, task_id: str) -> Path | None:
    """解析 task 对应的最新本地 checkpoint；无候选或 cfg 缺字段时 None。"""
    agent_cfg = load_rl_cfg(task_id)
    experiment_name = getattr(agent_cfg, "experiment_name", None)
    run_name = getattr(agent_cfg, "run_name", None)
    load_run = getattr(agent_cfg, "load_run", None)
    if not (experiment_name and run_name and load_run):
        # 非 velocity 类 runner cfg（如 tracking 任务）没有这些字段 → 不接管。
        return None
    log_dir = Path(log_root) / experiment_name
    if not log_dir.is_dir():
        return None
    runs = sorted(
        d for d in log_dir.iterdir() if d.is_dir() and re.match(load_run, d.name)
    )
    if not runs:
        return None
    # 目录名前缀为定宽时间戳（YYYY-MM-DD_HH-MM-SS），字典序 = 时间序；
    # 跳过只含 git / events 目录而无 checkpoint 的 run。
    for run_dir in reversed(runs):
        ckpt = _latest_checkpoint_by_step(run_dir)
        if ckpt is not None:
            return ckpt
    return None


def main() -> None:
    import mjlab.tasks  # noqa: F401  # 填充任务注册表（含本仓库 entry-point 包）
    import mjlab.scripts.play as play_script
    from mjlab.scripts._cli import maybe_print_top_level_help

    maybe_print_top_level_help("play")
    tasks = set(list_tasks())

    # 两段式解析复用 mjlab 原实现：这里只负责在 argv 中注入默认 checkpoint。
    if len(sys.argv) >= 2 and sys.argv[1] in tasks:
        task_id, rest = sys.argv[1], sys.argv[2:]
        explicit_ckpt = any(
            a == "--checkpoint-file" or a.startswith("--checkpoint-file=") for a in rest
        )
        explicit_wandb = any(
            a == "--wandb-run-path" or a.startswith("--wandb-run-path=") for a in rest
        )
        if not explicit_ckpt and not explicit_wandb:
            log_root = _argv_flag_value("--log-root") or "logs/rsl_rl"
            try:
                ckpt = resolve_latest_checkpoint(log_root, task_id)
            except Exception as exc:  # noqa: BLE001  # 配置缺字段等：不接管
                print(f"[INFO] 默认 checkpoint 解析失败（{exc}），跳过自动注入。")
                ckpt = None
            if ckpt is not None:
                print(
                    f"[INFO] 未指定 --checkpoint-file：使用默认最新 checkpoint {ckpt}"
                )
                sys.argv = [sys.argv[0], task_id, "--checkpoint-file", str(ckpt), *rest]
            else:
                print(
                    "[INFO] 未找到可自动注入的最新 checkpoint，"
                    "由 mjlab 处理原逻辑（未提供 --checkpoint-file 将报错）。"
                )
    play_script.main()
