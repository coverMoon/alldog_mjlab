"""兼容入口：委托 `alldog_mjlab.utils.mjwarp_epa` 的共享 EPA 启动实现。

保留本文件是为了兼容既有脚本/测试对
`tools/train_with_mjwarp_epa_patch.py` 的引用。项目级推荐入口现在是
`uv run train`（`alldog_mjlab.utils.train:main`），两者复用同一份实现（本文件不再
包含任何补丁代码）。

用法保持不变：
    uv run python tools/train_with_mjwarp_epa_patch.py <task-id> [...]
回退（原版 MjWarp）：
    MJWARP_EPA_PATCH=0 uv run python tools/train_with_mjwarp_epa_patch.py <task-id> [...]
"""
from __future__ import annotations

from alldog_mjlab.utils.mjwarp_epa import (  # noqa: F401
    EXPECTED_MJWARP_VERSION,
    PATCH_ID,
    _flag,
    _warnings_suppressed,
    install,
    launch,
    suppress_ls_iterations_warning,
    suppress_terminal_warnings,
)


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
