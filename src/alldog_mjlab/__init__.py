"""alldog_mjlab 包。

任务子包（`alldog_mjlab.tasks`）改为**延迟导入**：`import alldog_mjlab` 不再自动
导入 tasks，从而避免在 EPA 补丁安装之前经 MjLab 仿真模块提前加载 `mujoco_warp`。

任务注册仍由 MjLab 的 entry point 保证：`pyproject.toml` 的
`[project.entry-points."mjlab.tasks"]` 指向 `alldog_mjlab.tasks`，MjLab 启动时
`entry_point.load()` 会显式导入该子包并完成 `register_mjlab_task`。
`alldog_mjlab.tasks` 仍可通过属性访问或 `from alldog_mjlab import tasks` 获得。
"""

from importlib import import_module

__all__ = ["tasks"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "tasks":
        return import_module(".tasks", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
