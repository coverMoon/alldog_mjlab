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

环境变量：
    MJWARP_EPA_PATCH=0             关闭 EPA 补丁（本启动器退化为原版 + 下述 LS 处理）
    MJWARP_SUPPRESS_LS_WARNING=0   保留 LS_ITERATIONS 终端打印（默认在使用本启动器时关闭它）

加载机制（fail-closed）：
    1. 定位当前 venv 已安装的 `mujoco_warp`，要求版本 == 3.14.0，否则报错退出；
    2. **校验原始 `_src/collision_gjk.py` 的 SHA-256** 等于已核对的 3.14.0 原始哈希，
       否则报错退出（不自动接受任意哈希，防止上游源码被改动后静默套用补丁）；
    3. 在内存中施加两处**已 Review 的文本补丁**（每个 anchor 必须恰好匹配 1 次），
       并校验打补丁后的文件 SHA-256 等于预期值，否则报错退出；
    4. 缓存采用**内容寻址的不可变目录**：`CACHE_ROOT/mw<版本>_<打补丁后哈希前 16 位>`。
       已存在则**校验其内容**（文件哈希 + 关键文件存在）后才复用；不存在则在私有
       `mkdtemp` 目录内整包复制并打补丁，校验通过后用 `os.rename` **原子发布**。
       全程不删除共享目录，因此并发启动的训练进程不会互相破坏源码；
    5. 把缓存目录插到 sys.path[0] 后再 import `mujoco_warp`，因此整个包
       （含 `collision_convex.ccd_kernel_builder`）都来自副本；Warp 在**首次 build
       kernel** 时按复制后的源码解析 `wp.func` 调用 → CUDA kernel 使用补丁实现；
    6. import 后校验 `mujoco_warp.__file__` 指向缓存目录、`collision_gjk` 含新符号
       且该文件 SHA-256 与预期一致，否则报错退出。

    不修改 `.venv`、`pyproject.toml`、`uv.lock`、MjLab 或 Wolf MJCF/PD/DR/reward。

    关于 SHA-256：这里的哈希仅用于**内容完整性/版本钉死**（识别“被替换或损坏的
    源码副本”），不是安全机制，也不用于任何运行时契约。

LS_ITERATIONS 打印抑制（`suppress_ls_iterations_warning`）：
    MjWarp 的 solver 每步都用 `wp.atomic_or` 记录 LS_ITERATIONS 溢出（结构化统计与
    `overflow_ls_iterations` 指标不受影响），仅在 kernel build 时用
    `wp.static(warn_overflow & LS_ITERATIONS)` 决定是否 `wp.printf` 刷屏。本函数在
    **首次 kernel build 之前**（包 `Simulation._finish_init`）清除该位，因此只关掉
    终端打印，不改变求解器数值、`Data.overflow` 统计或任何训练参数。

补丁语义：`_epa()` 在 horizon 新增 face 时，只有“距离在 [lower2,upper2] 且投影不极端
偏离”的 face 才计入 `nvalid`；偏离的 face 置 invalid（仍留在多面体参与 horizon/可见性
更新，仅被选面跳过）。

注意：这是**数值稳定性修复**，仍有未验证的退化工况，不宣称已解决所有 EPA 问题。
本补丁作用于共享 EPA 路径，会同时影响使用同一算法的所有碰撞对（含 Black/BlackW）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

PATCH_ID = "epa-degenerate-face v1"
EXPECTED_MJWARP_VERSION = "3.14.0"
CACHE_ROOT = Path("/tmp/mjlab/mjwarp_epa_patch_cache")
_REL_COLLISION_GJK = Path("_src/collision_gjk.py")

# 缓存复用前必须存在的最小文件集合（collision_gjk.py 另行做哈希校验）。
_REQUIRED_FILES = (
    Path("__init__.py"),
    _REL_COLLISION_GJK,
    Path("_src/collision_convex.py"),
    Path("_src/io.py"),
)

# 已核对的原始文件哈希（两个独立来源一致：venv 安装副本 与 上游 3.15.0 wheel 中
# 该文件；二者 byte-identical，且 collision_gjk.py 在 3.14.0/3.15.0 间无差异）。
EXPECTED_SRC_SHA256 = "c288f8f37e4b7f8f9e0e48a92a46c3eee0d0efc6dd7fac65bba2522e11f39e48"
# 施加下述两处补丁后的预期哈希（由已 Review 的 tests/patch_mjwarp_epa_degenerate_face.diff）。
EXPECTED_PATCHED_SHA256 = "648a62a078794b79062fec83e4ad20166158f9c9b32367bf28e511daa05dae9b"

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _build_patched_text(original_text: str) -> str:
    """在内存中施加补丁并校验 anchor 唯一性；返回打补丁后的全文。"""
    if original_text.count(_ANCHOR_EPA) != 1:
        _fail(f"anchor `_epa` 匹配 {original_text.count(_ANCHOR_EPA)} 次（期望 1），源码版本不匹配")
    if original_text.count(_OLD_COND) != 1:
        _fail(f"anchor horizon-condition 匹配 {original_text.count(_OLD_COND)} 次（期望 1）")
    text = original_text.replace(_ANCHOR_EPA, _HELPER_BLOCK + _ANCHOR_EPA, 1)
    text = text.replace(_OLD_COND, _NEW_COND, 1)
    if "_face_degenerate_projection" not in text:
        _fail("补丁结果中未找到 `_face_degenerate_projection`")
    got = hashlib.sha256(text.encode()).hexdigest()
    if got != EXPECTED_PATCHED_SHA256:
        _fail(f"打补丁后内容哈希 {got} != 预期 {EXPECTED_PATCHED_SHA256}（补丁文本与预期不符）")
    return text


def _cache_dir_for_patched_hash() -> Path:
    return CACHE_ROOT / f"mw{EXPECTED_MJWARP_VERSION}_{EXPECTED_PATCHED_SHA256[:16]}"


def _cache_is_valid(cache_dir: Path) -> bool:
    """仅当缓存目录**内容**正确时才认为可复用（不依赖路径存在）。"""
    pkg = cache_dir / "mujoco_warp"
    if not all((pkg / rel).is_file() for rel in _REQUIRED_FILES):
        return False
    try:
        return _sha256_file(pkg / _REL_COLLISION_GJK) == EXPECTED_PATCHED_SHA256
    except OSError:
        return False


def _populate_cache_dir(target: Path, src_pkg: Path, patched_text: str) -> None:
    """把已安装包复制进 target/mujoco_warp 并写入打补丁后的 collision_gjk.py。"""
    shutil.copytree(
        src_pkg,
        target / "mujoco_warp",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (target / "mujoco_warp" / _REL_COLLISION_GJK).write_text(patched_text)


def _publish_cache(target: Path, src_pkg: Path, patched_text: str, verbose: bool) -> Path:
    """确保 `target` 成为内容正确的缓存目录；返回本次实际使用的目录。

    已存在且校验通过则直接复用；否则在私有 `mkdtemp` 目录内构建、校验后用
    `os.rename` 原子发布。**从不删除共享目录**，因此并发进程不会互相破坏源码。
    """
    if _cache_is_valid(target):
        if verbose:
            print(f"[mjwarp-epa-patch] cache: reuse {target}")
        return target

    tmp_dir = Path(tempfile.mkdtemp(dir=str(CACHE_ROOT), prefix=".tmp-"))
    try:
        _populate_cache_dir(tmp_dir, src_pkg, patched_text)
        if not _cache_is_valid(tmp_dir):
            _fail("私有构建目录校验失败（复制或写入结果与预期补丁不一致）")
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    try:
        os.rename(tmp_dir, target)  # 原子发布；目标为非空目录时抛 OSError
        if verbose:
            print(f"[mjwarp-epa-patch] cache: built {target}")
        return target
    except OSError:
        if _cache_is_valid(target):  # 并发竞争：其他进程先发布了
            shutil.rmtree(tmp_dir, ignore_errors=True)  # 自己的私有目录
            if verbose:
                print(f"[mjwarp-epa-patch] cache: reuse (race) {target}")
            return target
        if verbose:
            print(f"[mjwarp-epa-patch] cache: private {tmp_dir}（{target} 被占用且校验不通过）")
        return tmp_dir


def install(verbose: bool = True) -> Path:
    """把已安装的 mujoco_warp 复制到内容寻址的不可变缓存目录、打补丁并在 sys.path[0] 启用。

    并发安全：只在私有 `mkdtemp` 目录内写入，校验通过后 `os.rename` 原子发布；
    永不删除共享缓存目录，因此多个训练进程同时启动不会互相破坏源码。
    重复调用安全（若已从缓存目录导入则直接返回）。
    fail-closed：任何一步失败抛 SystemExit，绝不静默回退到原版实现。
    """
    existing = sys.modules.get("mujoco_warp")
    if existing is not None:
        pkg_dir = Path(getattr(existing, "__file__", "") or "").resolve().parent
        gjk = pkg_dir / _REL_COLLISION_GJK
        if (
            CACHE_ROOT.resolve() in pkg_dir.parents
            and gjk.is_file()
            and _sha256_file(gjk) == EXPECTED_PATCHED_SHA256
        ):
            if verbose:
                print(f"[mjwarp-epa-patch] 已加载（{PATCH_ID}）: {pkg_dir}")
            return pkg_dir.parent
        _fail(
            "mujoco_warp 已在安装补丁之前被导入（加载方式错误）；"
            "请在 import mjlab 之前调用 install()"
        )

    ver = _installed_version()
    if ver != EXPECTED_MJWARP_VERSION:
        _fail(f"mujoco-warp 版本 {ver} != 期望 {EXPECTED_MJWARP_VERSION}（版本校验失败）")

    src_pkg = _installed_pkg_dir()
    src_gjk = src_pkg / _REL_COLLISION_GJK
    src_sha = _sha256_file(src_gjk)
    if src_sha != EXPECTED_SRC_SHA256:
        _fail(
            f"原始 {_REL_COLLISION_GJK} 哈希 {src_sha} != 已核对哈希 {EXPECTED_SRC_SHA256}"
            f"（来源 {src_gjk}）"
        )

    patched_text = _build_patched_text(src_gjk.read_text())

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    final_dir = _cache_dir_for_patched_hash()
    target = final_dir
    if target.exists() and not _cache_is_valid(target):
        # 目标存在但内容不正确：不删除共享目录（可能正被其他进程使用），改用固定的
        # 后备目录，避免每次运行都新建（累积目录数有界）。
        target = final_dir.with_name(final_dir.name + ".fallback")
    cache_dir = _publish_cache(target, src_pkg, patched_text, verbose)

    sys.path.insert(0, str(cache_dir))
    import mujoco_warp  # noqa: F401

    mw_file = Path(mujoco_warp.__file__).resolve()
    if cache_dir.resolve() not in mw_file.parents:
        _fail(f"import 解析到 {mw_file}，不是缓存目录 {cache_dir}")
    from mujoco_warp._src import collision_gjk

    if not hasattr(collision_gjk, "_face_degenerate_projection"):
        _fail("collision_gjk 缺少 `_face_degenerate_projection`（补丁未生效）")
    loaded_sha = _sha256_file(Path(collision_gjk.__file__))
    if loaded_sha != EXPECTED_PATCHED_SHA256:
        _fail(f"已加载 collision_gjk.py 哈希 {loaded_sha} != 预期 {EXPECTED_PATCHED_SHA256}")

    if verbose:
        print(
            "[mjwarp-epa-patch] ENABLED\n"
            f"  patch        : {PATCH_ID}\n"
            f"  base version : mujoco-warp {ver} (src sha256 {src_sha[:16]}…)\n"
            f"  patched sha  : {EXPECTED_PATCHED_SHA256[:16]}…\n"
            f"  load from    : {mw_file}\n"
            f"  cache dir    : {cache_dir}"
        )
    return cache_dir


def suppress_ls_iterations_warning(verbose: bool = True) -> None:
    """在首次 kernel build 之前清除 `opt.warn_overflow` 的 LS_ITERATIONS 位。

    只关闭 MjWarp 的终端刷屏打印；`Data.overflow` 记录与 `overflow_ls_iterations`
    指标不受影响（`wp.atomic_or` 与打印开关相互独立）。不改求解器/PD/DR/训练参数。
    """
    import mujoco_warp

    from mjlab.sim import Simulation

    ls_bit = int(mujoco_warp.OverflowType.LS_ITERATIONS)
    original_finish_init = Simulation._finish_init
    state = {"reported": False}

    def patched_finish_init(self):  # type: ignore[no-untyped-def]
        before = int(self._wp_model.opt.warn_overflow)
        self._wp_model.opt.warn_overflow = before & ~ls_bit
        if verbose and not state["reported"]:
            state["reported"] = True
            print(
                "[mjwarp-ls-warning] LS_ITERATIONS 打印已关闭（仅打印，不改数值/统计）\n"
                f"  warn_overflow: {before} -> {int(self._wp_model.opt.warn_overflow)}"
                f" (cleared bit {ls_bit})"
            )
        return original_finish_init(self)

    Simulation._finish_init = patched_finish_init


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "off", "no")


def main() -> None:
    if _flag("MJWARP_EPA_PATCH", "1"):
        install(verbose=True)
    else:
        print("[mjwarp-epa-patch] DISABLED via MJWARP_EPA_PATCH=0 (native mujoco_warp)")

    if _flag("MJWARP_SUPPRESS_LS_WARNING", "1"):
        suppress_ls_iterations_warning(verbose=True)
    else:
        print("[mjwarp-ls-warning] KEPT via MJWARP_SUPPRESS_LS_WARNING=0")

    from mjlab.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
