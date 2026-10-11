"""MjWarp 3.14 EPA 异常投影过滤补丁：隔离加载器 + Wolf 训练启动器（默认关闭）。

只在**通过本脚本启动**时才加载补丁；原生 `uv run train ...` 完全不变。
本脚本负责：加载补丁 → 委托原生 `mjlab.scripts.train.main()`（不复制任何训练逻辑）。

用法（**启用**补丁）：
    uv run python tools/train_with_mjwarp_epa_patch.py wolf-rough-him \
        --env.scene.num-envs 3000 --agent.resume True \
        --agent.load-run '2026-10-09_22-45-14_wolf_flat_him'

用法（**关闭**/回退原版）：
    uv run train wolf-rough-him ...            # 原生入口，从不加载补丁
    MJWARP_EPA_PATCH=0 uv run python tools/train_with_mjwarp_epa_patch.py ...  # 显式关

加载机制（fail-closed）：
    1. 定位当前 venv 已安装的 `mujoco_warp`，要求版本 == 3.14.0，否则报错退出；
    2. 整包复制到隔离目录 CACHE_DIR（排除 __pycache__）；
    3. 对副本 `_src/collision_gjk.py` 施加两处**已 Review 的文本补丁**；每个 anchor
       必须恰好匹配一次，否则报错退出；
    4. 把 CACHE_DIR 插到 sys.path[0] 后再 import `mujoco_warp`，因此整个包
       （含 `collision_convex.ccd_kernel_builder`）都来自副本；Warp 在**首次 build
       kernel** 时按复制后的源码解析 `wp.func` 调用 → CUDA kernel 使用补丁实现；
    5. import 后校验 `mujoco_warp.__file__` 指向副本、且 `collision_gjk` 含新符号，
       否则报错退出。

    不修改 `.venv`、`pyproject.toml`、`uv.lock`、MjLab 或 Wolf MJCF/PD/DR/reward。

补丁语义：`_epa()` 在 horizon 新增 face 时，只有“距离在 [lower2,upper2] 且投影不极端
偏离”的 face 才计入 `nvalid`；偏离的 face 置 invalid（仍留在多面体参与 horizon/可见性
更新，仅被选面跳过）。

注意：这是**数值稳定性修复**，仍有未验证的退化工况，不宣称已解决所有 EPA 问题。
本补丁作用于共享 EPA 路径，会同时影响使用同一算法的所有碰撞对（含 Black/BlackW）。
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

PATCH_ID = "epa-degenerate-face v1"
EXPECTED_MJWARP_VERSION = "3.14.0"
CACHE_DIR = Path("/tmp/mjlab/mjwarp_epa_patch_3.14.0_v1")
_REL_COLLISION_GJK = Path("_src/collision_gjk.py")

# --- 已 Review 的补丁文本（与 tests/patch_mjwarp_epa_degenerate_face.diff 一致） ---
_ANCHOR_EPA = "@wp.func\ndef _epa("
_HELPER_BLOCK = '''@wp.func
def _face_degenerate_projection(pt: Polytope, idx: int) -> bool:
  """Conservative magnitude filter for an extremely deviating face projection.

  ``_project_origin_plane`` computes the origin's projection ``face_pr`` onto a face
  plane. For a usable (well-shaped) support face this projection lies close to the face,
  so its distance to the farthest face vertex is on the order of the longest edge. A
  near-degenerate (needle) triangle can instead produce a plane whose projection sits far
  away from the triangle while still reporting a spuriously small plane distance, and its
  barycentric witness is then ill-conditioned. This is NOT an "inside the triangle"
  geometric claim: it is a conservative test that flags only extreme projection deviation
  ((distance to farthest vertex) > 2 * (longest edge)). Scale-aware (compares squared
  lengths), so no absolute threshold is assumed.
  """
  face = _get_face_verts(pt.face[idx])
  p1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]
  p2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]
  p3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]
  r = pt.face_pr[idx]
  e12 = wp.dot(p1 - p2, p1 - p2)
  e13 = wp.dot(p1 - p3, p1 - p3)
  e23 = wp.dot(p2 - p3, p2 - p3)
  emax2 = wp.max(e12, wp.max(e13, e23))
  dfar2 = wp.max(
    wp.dot(r - p1, r - p1), wp.max(wp.dot(r - p2, r - p2), wp.dot(r - p3, r - p3))
  )
  return dfar2 > 4.0 * emax2


'''

_OLD_COND = """      pt.nface += 1

      if dist2 >= lower2 and dist2 <= upper2:
        nvalid += 1
      else:
        pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])"""

_NEW_COND = """      pt.nface += 1

      # A face whose projection deviates extremely is marked invalid (kept in the
      # polytope for horizon/visibility, skipped by selection). It must NOT count
      # toward nvalid, otherwise nvalid disagrees with the selectable-face set.
      if (
        dist2 >= lower2
        and dist2 <= upper2
        and not _face_degenerate_projection(pt, pt.nface - 1)
      ):
        nvalid += 1
      else:
        pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])"""


def _fail(msg: str) -> None:
    raise SystemExit(f"[mjwarp-epa-patch] FAIL-CLOSED: {msg}")


def _installed_pkg_dir() -> Path:
    # find_spec for a top-level name does not import the package.
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or not spec.submodule_search_locations:
        _fail("当前环境找不到 mujoco_warp")
    return Path(list(spec.submodule_search_locations)[0])


def _installed_version() -> str:
    import importlib.metadata as md

    try:
        return md.version("mujoco-warp")
    except Exception as exc:  # noqa: BLE001
        _fail(f"读取 mujoco-warp 版本失败: {exc}")
        raise  # pragma: no cover


def _apply_patch_to_file(path: Path) -> None:
    text = path.read_text()
    if text.count(_ANCHOR_EPA) != 1:
        _fail(f"anchor `_epa` 匹配 {text.count(_ANCHOR_EPA)} 次（期望 1），源码版本不匹配")
    if text.count(_OLD_COND) != 1:
        _fail(f"anchor horizon-condition 匹配 {text.count(_OLD_COND)} 次（期望 1）")
    text = text.replace(_ANCHOR_EPA, _HELPER_BLOCK + _ANCHOR_EPA, 1)
    text = text.replace(_OLD_COND, _NEW_COND, 1)
    path.write_text(text)
    if "_face_degenerate_projection" not in path.read_text():
        _fail("补丁写入后未找到 `_face_degenerate_projection`")


def install(verbose: bool = True) -> Path:
    """把已安装的 mujoco_warp 复制到隔离目录、打补丁并置于 sys.path[0]。

    重复调用安全（若已从 CACHE_DIR 导入则直接返回）。fail-closed：任何一步失败抛
    SystemExit，绝不静默回退到原版实现。
    """
    already = sys.modules.get("mujoco_warp")
    if already is not None:
        f = Path(getattr(already, "__file__", ""))
        if CACHE_DIR in f.parents:
            if verbose:
                print(f"[mjwarp-epa-patch] 已加载（{PATCH_ID}）: {f}")
            return CACHE_DIR
        _fail(
            "mujoco_warp 已在安装补丁之前被导入（加载方式错误）；"
            "请在 import mjlab 之前调用 install()"
        )

    ver = _installed_version()
    if ver != EXPECTED_MJWARP_VERSION:
        _fail(f"mujoco-warp 版本 {ver} != 期望 {EXPECTED_MJWARP_VERSION}（版本校验失败）")

    src_pkg = _installed_pkg_dir()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst_pkg = CACHE_DIR / "mujoco_warp"
    shutil.copytree(src_pkg, dst_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _apply_patch_to_file(dst_pkg / _REL_COLLISION_GJK)

    sys.path.insert(0, str(CACHE_DIR))
    import mujoco_warp  # noqa: F401

    mw_file = Path(mujoco_warp.__file__).resolve()
    if CACHE_DIR.resolve() not in mw_file.parents:
        _fail(f"import 解析到 {mw_file}，不是隔离副本 {CACHE_DIR}")
    from mujoco_warp._src import collision_gjk

    if not hasattr(collision_gjk, "_face_degenerate_projection"):
        _fail("collision_gjk 缺少 `_face_degenerate_projection`（补丁未生效）")

    if verbose:
        print(
            "[mjwarp-epa-patch] ENABLED\n"
            f"  patch        : {PATCH_ID}\n"
            f"  base version : mujoco-warp {ver} (from {src_pkg})\n"
            f"  load from    : {mw_file}\n"
            f"  cache dir    : {CACHE_DIR}"
        )
    return CACHE_DIR


def main() -> None:
    import os

    flag = os.environ.get("MJWARP_EPA_PATCH", "1").strip().lower()
    if flag in ("0", "false", "off", "no"):
        print("[mjwarp-epa-patch] DISABLED via MJWARP_EPA_PATCH=0 (native mujoco_warp)")
    else:
        install(verbose=True)
    from mjlab.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
