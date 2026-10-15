"""MjWarp 3.14 EPA 退化面过滤补丁（v1 投影过滤 + v2 needle 形状过滤）的共享实现（唯一权威来源）。

两处入口复用本模块：
  - 项目级 console script `uv run train`（`alldog_mjlab.utils.train:main` -> launch()）
  - 兼容入口 `tools/train_with_mjwarp_epa_patch.py`（-> launch()）

补丁内容（两处文本替换，见 `_HELPER_BLOCK` / `_NEW_COND`）：
  v1 `_face_degenerate_projection`：投影 "极端偏离" 过滤（面平面上的原点投影远离三角形，
      判据 dfar2 > 4 * emax2）。修复 §37 的 needle 面伪接触。
  v2 `_face_needle_shape`：尺度无关的 needle/sliver 形状过滤（最小高度相对最长边过小，
      判据 nn < 1e-8 * emax2 * emax2）。修复 env-1854 的 needle 面伪接触（v1 的判据
      在最长边很长时过于宽松，漏掉该种退化）。
  两者在同一 horizon 新增面判定处取并：不满足任一条的 face 置 invalid（仍留在 polytope
  参与 horizon / 可见性维护，仅不参与选面），`nvalid` 计数与可选面集合保持一致。

加载机制（fail-closed，细节见各常量/函数 docstring）：版本校验 -> 原始 SHA-256 校验
-> 内存打补丁 -> 打补丁后 SHA-256 校验 -> 内容寻址缓存/原子发布 -> sys.path 前置后
import -> import 后路径与两处 guard 符号校验。任何一步失败 SystemExit，绝不静默回退
未修补实现。

环境变量：
  MJWARP_EPA_PATCH=0           关闭 EPA 补丁（使用已安装的原版 mujoco_warp）
  MJWARP_SUPPRESS_WARNINGS=0   保留 LS_ITERATIONS/HFIELD/ITERATIONS 终端打印
  MJWARP_SUPPRESS_LS_WARNING=0 旧名兼容开关（设置时优先于统一开关）

本模块顶层只 import 标准库；`mujoco_warp` 与 `mjlab` 仅在函数内部、且补丁安装完成后
才 import。因此 import 本模块本身不会提前加载 `mujoco_warp`。
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

PATCH_ID = "epa-degenerate-face v2"
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
# 施加 v1(投影) + v2(needle 形状) 两处补丁后的预期哈希。
# 由本模块 `_HELPER_BLOCK` / `_NEW_COND` 对已核对原始文本施加后计算（见 _build_patched_text）。
# 与 v1 补丁哈希 (648a62a0…) 不同：内容寻址缓存目录随之变化，旧 v1 缓存不会被复用。
EXPECTED_PATCHED_SHA256 = "dfefeac6ea5d17154a70213925c0db277a11bc54d374bbac67b50166e3bbde69"

# --- 已 Review 的补丁文本（v1 投影过滤 + v2 needle 形状过滤） ---
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


@wp.func
def _face_needle_shape(pt: Polytope, idx: int) -> bool:
  """Scale-free needle/sliver filter for a candidate support face.

  With ``n = (p2 - p1) x (p3 - p1)`` we have ``|n| = 2 * area`` and, for the longest
  edge ``emax``, the minimal altitude is ``hmin = 2 * area / emax``. Hence
  ``nn / emax2 == hmin**2`` (``nn = |n|**2``, ``emax2 = emax**2``). A face whose minimal
  altitude is negligible compared with its own longest edge makes ``_tri_affine_coord``
  ill-conditioned (it divides by ``M_max`` ~ 2 * area), so ``_epa_witness`` can
  interpolate witnesses hundreds of metres away from both geoms. The test below uses a
  dimensionless aspect ratio (``hmin / emax < 1e-4``, i.e. ``nn < 1e-8 * emax2**2``), so
  no absolute length threshold is assumed; it is independent of the v1 projection test,
  which is too permissive when the longest edge is much longer than the face is thick.
  A fully degenerate zero-length triangle (``emax2 <= 0``) is rejected as well, matching
  the float32 arithmetic of ``_attach_face`` / ``_face_degenerate_projection``.
  """
  face = _get_face_verts(pt.face[idx])
  p1 = pt.vert[2 * face[0]] - pt.vert[2 * face[0] + 1]
  p2 = pt.vert[2 * face[1]] - pt.vert[2 * face[1] + 1]
  p3 = pt.vert[2 * face[2]] - pt.vert[2 * face[2] + 1]
  e12 = wp.dot(p1 - p2, p1 - p2)
  e13 = wp.dot(p1 - p3, p1 - p3)
  e23 = wp.dot(p2 - p3, p2 - p3)
  emax2 = wp.max(e12, wp.max(e13, e23))
  if emax2 <= 0.0:
    return True
  n = wp.cross(p2 - p1, p3 - p1)
  nn = wp.dot(n, n)
  return nn < 1e-8 * emax2 * emax2


'''

_OLD_COND = """      pt.nface += 1

      if dist2 >= lower2 and dist2 <= upper2:
        nvalid += 1
      else:
        pt.face[pt.nface - 1] = _invalidate_face(pt.face[pt.nface - 1])"""

_NEW_COND = """      pt.nface += 1

      # A face whose projection deviates extremely (v1), or whose triangle is a
      # near-degenerate needle/sliver (v2), is marked invalid (kept in the polytope for
      # horizon/visibility, skipped by selection). It must NOT count toward nvalid,
      # otherwise nvalid disagrees with the selectable-face set.
      if (
        dist2 >= lower2
        and dist2 <= upper2
        and not _face_degenerate_projection(pt, pt.nface - 1)
        and not _face_needle_shape(pt, pt.nface - 1)
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
    for symbol in ("_face_degenerate_projection", "_face_needle_shape"):
        if symbol not in text:
            _fail(f"补丁结果中未找到 `{symbol}`")
    if "not _face_needle_shape(pt, pt.nface - 1)" not in text:
        _fail("补丁结果中 horizon 新增面判定未引用 v2 形状守卫")
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
        _fail("collision_gjk 缺少 `_face_degenerate_projection`（v1 补丁未生效）")
    if not hasattr(collision_gjk, "_face_needle_shape"):
        _fail("collision_gjk 缺少 `_face_needle_shape`（v2 补丁未生效）")
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


# 仅关闭终端刷屏打印；这些位仍会被无条件记录到 Data.overflow（指标不受影响）。
_SUPPRESSED_WARNING_TYPES = ("LS_ITERATIONS", "HFIELD", "ITERATIONS")

_warning_patch_installed = False


def suppress_terminal_warnings(verbose: bool = True) -> None:
    """在首次 kernel build 之前清除 `opt.warn_overflow` 中三类刷屏打印位。

    清除：`LS_ITERATIONS | HFIELD | ITERATIONS`。
    只关闭 MjWarp 的终端打印；`Data.overflow` 记录及对应 overflow_* 指标不受影响
    （`wp.atomic_or` / `overflow_out |=` 与打印开关相互独立）。不改 solver
    iterations / ls_iterations / hfield resolution / 碰撞容量等任何参数。
    """
    global _warning_patch_installed
    if _warning_patch_installed:
        return

    import mujoco_warp

    from mjlab.sim import Simulation

    overflow = mujoco_warp.OverflowType
    cleared_mask = 0
    for type_name in _SUPPRESSED_WARNING_TYPES:
        cleared_mask |= int(getattr(overflow, type_name))
    cleared_names = " | ".join(_SUPPRESSED_WARNING_TYPES)

    original_finish_init = Simulation._finish_init
    state = {"reported": False}

    def patched_finish_init(self):  # type: ignore[no-untyped-def]
        before = int(self._wp_model.opt.warn_overflow)
        self._wp_model.opt.warn_overflow = before & ~cleared_mask
        if verbose and not state["reported"]:
            state["reported"] = True
            print(
                f"[mjwarp-warning] 终端刷屏打印已关闭：{cleared_names}"
                "（仅打印，不改数值/统计）\n"
                f"  warn_overflow: {before} -> {int(self._wp_model.opt.warn_overflow)}"
                f" (cleared mask {cleared_mask})"
            )
        return original_finish_init(self)

    Simulation._finish_init = patched_finish_init
    _warning_patch_installed = True


def suppress_ls_iterations_warning(verbose: bool = True) -> None:
    """兼容旧名（历史版本只清 LS_ITERATIONS）：等价于 `suppress_terminal_warnings`。"""
    suppress_terminal_warnings(verbose=verbose)


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "off", "no")


def _warnings_suppressed() -> tuple[bool, str]:
    """统一开关：MJWARP_SUPPRESS_WARNINGS（默认开）；旧名设置时优先。"""
    legacy = "MJWARP_SUPPRESS_LS_WARNING"
    if os.environ.get(legacy) is not None:
        return _flag(legacy, "1"), legacy
    unified = "MJWARP_SUPPRESS_WARNINGS"
    return _flag(unified, "1"), unified


def launch() -> None:
    """共享启动序列：按环境变量安装补丁、配置终端 warning，再进入 MjLab 原生训练。

    不复制训练逻辑、不使用 subprocess。`mujoco_warp` 只在 install() /
    suppress_terminal_warnings() 内、顺序安装之后才被导入。
    """
    if _flag("MJWARP_EPA_PATCH", "1"):
        install(verbose=True)
    else:
        print("[mjwarp-epa-patch] DISABLED via MJWARP_EPA_PATCH=0 (native mujoco_warp)")

    suppress, source = _warnings_suppressed()
    if suppress:
        suppress_terminal_warnings(verbose=True)
    else:
        print(f"[mjwarp-warning] KEPT via {source}=0（保留 LS/HFIELD/ITERATIONS 打印）")

    from mjlab.scripts.train import main as train_main

    train_main()
