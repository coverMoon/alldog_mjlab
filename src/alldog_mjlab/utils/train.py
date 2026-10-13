"""项目级训练入口（console script `uv run train`）。

启动顺序（保证 EPA 补丁在 `mujoco_warp` / MjLab 仿真模块之前安装）：

    uv run train ...
      -> alldog_mjlab.utils.train.main()
        -> alldog_mjlab.utils.mjwarp_epa.launch()
          -> 读取 MJWARP_EPA_PATCH，安装补丁或选择原版
          -> 读取 MJWARP_SUPPRESS_WARNINGS，注册终端 warning 控制
          -> import mjlab.scripts.train.main
          -> 调用原生 MjLab train_main()

不复制 MjLab train CLI，不使用 subprocess，不新建训练循环。本模块顶层只 import
本项目的轻量模块；`mujoco_warp` / `mjlab` 仅在 launch() 内部、补丁安装后导入。
"""
from __future__ import annotations


def main() -> None:
    from alldog_mjlab.utils.mjwarp_epa import launch

    launch()


if __name__ == "__main__":
    main()
