"""Wolf rough 的 terrain 数学与 generator 组装（Wolf task local）。

实现方式与 black/terrain.py 的原创工作独立同构（数值相同、代码归属于 Wolf，
不 import Black）：

    wolf_config.py terrain 数值（patch 尺寸、难度、各 terrain 比例、noise 幅值）
    env_cfgs.py   terrain 装配（terrain_type / max_init_terrain_level / curriculum term）
    本文件        terrain 数学（task-local rough slope primitive + generator 组装）

MjLab v1.6.0 没有「slope + noise」的 native composition：每个 native heightfield
terrain 都是「算一个 int16 高度场数组 → 建 hfield + geom」。因此本文件的
`WolfRoughSlopeTerrainCfg` 复用两个 native terrain 的数学（slope 部分与
`HfPyramidSlopedTerrainCfg` 的 `border_width=0` 分支一致，noise 部分与
`HfRandomUniformTerrainCfg` 的 `border_width=0` 分支一致），把两个 int16 数组相加，
再走与 native 相同的 hfield / material / spawn origin 构造（复用 native 的
`color_by_height`）。没有新增 terrain framework、没有第二套 generator。

这些数值是 Wolf 的首版候选（沿用 Black rough 已实现值），不宣称已验证适合
Wolf 的轮足运动；后续按训练表现调整。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy import interpolate

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from alldog_mjlab.tasks.velocity.wolf.wolf_config import WOLF_CONFIG, TerrainParams


@dataclass(kw_only=True)
class WolfRoughSlopeTerrainCfg(SubTerrainCfg):
    """pyramid slope + random uniform noise（Wolf rough slope）。

    难度语义（difficulty d ∈ [0, 0.9]）：

        slope     = slope_range[0] + d * (slope_range[1] - slope_range[0])
        amplitude = noise_base + noise_gain * d

    slope 部分是四向金字塔（中心为 flat platform，向外下降），噪声部分是
    ±amplitude 内、以 `noise_step` 量化、以 `noise_downsample` 间距采样再双线性
    插值的均匀噪声，噪声叠加在整个 patch 上（包含 spawn platform）。

    heightfield 保留 absolute zero（geom z offset = elevation_min ×
    vertical_scale），表面高度即 `noise × vertical_scale`，patch 边缘与相邻
    terrain / border 在 z = 0 衔接；spawn origin z 取 patch 中心 ±1 m 区域的
    最大 raw height。
    """

    slope_range: tuple[float, float] = field()
    noise_base: float = field()
    noise_gain: float = field()
    noise_step: float = field()
    noise_downsample: float = field()
    platform_width: float = field()
    horizontal_scale: float = field()
    vertical_scale: float = field()
    base_thickness_ratio: float = field()

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")

        noise = self._slope_noise(difficulty) + self._uniform_noise(difficulty, rng)

        elevation_min = int(np.min(noise))
        elevation_max = int(np.max(noise))
        elevation_range = (
            elevation_max - elevation_min if elevation_max != elevation_min else 1
        )

        # 高度场保持原始 absolute zero：geom 下移 elevation_min，使表面高度重新等于
        # ``noise * vertical_scale``（与 legacy ``height_field_raw`` 一致，patch 边缘
        # 回到 z = 0，与相邻 terrain / border 衔接）。
        hfield_z_offset = elevation_min * self.vertical_scale

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio

        if elevation_range > 0:
            normalized_elevation = (noise - elevation_min) / elevation_range
        else:
            normalized_elevation = np.zeros_like(noise)

        unique_id = uuid.uuid4().hex
        hfield = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[
                self.size[0] / 2,
                self.size[1] / 2,
                max_physical_height,
                base_thickness,
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
        )

        physical_heights = hfield_z_offset + normalized_elevation * max_physical_height
        material_name = color_by_height(spec, noise, unique_id, physical_heights)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=hfield.name,
            pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
            material=material_name,
        )

        # spawn origin z = patch 中心 ±1 m 区域的最大 raw terrain height。
        x1 = int((self.size[0] / 2 - 1.0) / self.horizontal_scale)
        x2 = int((self.size[0] / 2 + 1.0) / self.horizontal_scale)
        y1 = int((self.size[1] / 2 - 1.0) / self.horizontal_scale)
        y2 = int((self.size[1] / 2 + 1.0) / self.horizontal_scale)
        spawn_height = float(np.max(noise[x1:x2, y1:y2])) * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        if self.flat_patch_sampling is not None:
            raise NotImplementedError(
                "WolfRoughSlopeTerrainCfg 未实现 flat patch sampling"
            )

        geom = TerrainGeometry(geom=hfield_geom, hfield=hfield)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=None)

    def _slope_noise(self, difficulty: float) -> np.ndarray:
        """`HfPyramidSlopedTerrainCfg`（inverted=False, border_width=0）的 slope 高度场。"""
        slope = self.slope_range[0] + difficulty * (
            self.slope_range[1] - self.slope_range[0]
        )
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)

        height_max = int(slope * self.size[0] / 2 / self.vertical_scale)

        center_x = int(width_pixels / 2)
        center_y = int(length_pixels / 2)

        x = np.arange(0, width_pixels)
        y = np.arange(0, length_pixels)
        xx, yy = np.meshgrid(x, y, sparse=True)

        xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width_pixels, 1)
        yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length_pixels)

        hf_raw = height_max * xx * yy

        platform_pixels = int(self.platform_width / self.horizontal_scale / 2)
        x_pf = width_pixels // 2 - platform_pixels
        y_pf = length_pixels // 2 - platform_pixels
        z_pf = hf_raw[x_pf, y_pf]
        hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

        return np.rint(hf_raw).astype(np.int16)

    def _uniform_noise(self, difficulty: float, rng: np.random.Generator) -> np.ndarray:
        """`HfRandomUniformTerrainCfg`（border_width=0）的 ±amplitude 均匀噪声高度场。"""
        amplitude = self.noise_base + self.noise_gain * difficulty

        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        width_downsampled = int(self.size[0] / self.noise_downsample)
        length_downsampled = int(self.size[1] / self.noise_downsample)

        height_min = int(-amplitude / self.vertical_scale)
        height_max = int(amplitude / self.vertical_scale)
        height_step = int(self.noise_step / self.vertical_scale)

        height_range = np.arange(height_min, height_max + height_step, height_step)
        height_field_downsampled = rng.choice(
            height_range, size=(width_downsampled, length_downsampled)
        )

        x = np.linspace(0, self.size[0], width_downsampled)
        y = np.linspace(0, self.size[1], length_downsampled)
        # kx=ky=1（双线性）：MjLab native preset 用三次样条会在采样点之间
        # overshoot 超出 ±amplitude；双线性与 legacy `interp2d(kind='linear')` 一致。
        func = interpolate.RectBivariateSpline(x, y, height_field_downsampled, kx=1, ky=1)

        x_upsampled = np.linspace(0, self.size[0], width_pixels)
        y_upsampled = np.linspace(0, self.size[1], length_pixels)
        z_upsampled = func(x_upsampled, y_upsampled)

        return np.rint(z_upsampled).astype(np.int16)


def wolf_rough_terrain_generator_cfg(terrain: TerrainParams | None = None) -> TerrainGeneratorCfg:
    """Wolf rough v1 的 curriculum terrain generator（数值来自 WOLF_CONFIG.terrain）。

    7 类 sub-terrain（curriculum 模式下每类一列）、10 行难度；`proportion` 在
    curriculum 模式下是 env 分配权重，不是列数。

    stairs（up / down）使用 native box 金字塔台阶，step_height 按 difficulty
    线性插值（``base + gain * d``）；与 legacy heightfield 台缘的 0.1 m 网格量化
    差异是 framework difference（同 Black rough），不声称 exact reproduction。
    """
    terrain = WOLF_CONFIG.terrain if terrain is None else terrain
    terrain.validate()
    proportions = terrain.proportions
    return TerrainGeneratorCfg(
        curriculum=True,
        size=terrain.size,
        border_width=terrain.border_width,
        num_rows=terrain.num_rows,
        difficulty_range=terrain.difficulty_range,
        add_lights=True,
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=proportions["flat"]),
            "smooth_slope_up": HfPyramidSlopedTerrainCfg(
                proportion=proportions["smooth_slope_up"],
                slope_range=terrain.slope_range,
                platform_width=terrain.platform_width,
                horizontal_scale=terrain.horizontal_scale,
                vertical_scale=terrain.vertical_scale,
                inverted=False,
            ),
            "smooth_slope_down": HfPyramidSlopedTerrainCfg(
                proportion=proportions["smooth_slope_down"],
                slope_range=terrain.slope_range,
                platform_width=terrain.platform_width,
                horizontal_scale=terrain.horizontal_scale,
                vertical_scale=terrain.vertical_scale,
                inverted=True,
            ),
            "rough_slope": WolfRoughSlopeTerrainCfg(
                proportion=proportions["rough_slope"],
                slope_range=terrain.slope_range,
                noise_base=terrain.rough_noise_base,
                noise_gain=terrain.rough_noise_gain,
                noise_step=terrain.rough_noise_step,
                noise_downsample=terrain.rough_noise_downsample,
                base_thickness_ratio=terrain.rough_base_thickness_ratio,
                platform_width=terrain.platform_width,
                horizontal_scale=terrain.horizontal_scale,
                vertical_scale=terrain.vertical_scale,
            ),
            "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                proportion=proportions["discrete_obstacles"],
                obstacle_height_mode="choice",
                obstacle_width_range=terrain.obstacle_width_range,
                obstacle_height_range=terrain.obstacle_height_range,
                num_obstacles=terrain.obstacle_count,
                platform_width=terrain.platform_width,
                horizontal_scale=terrain.horizontal_scale,
                vertical_scale=terrain.vertical_scale,
                square_obstacles=False,
            ),
            "stairs_up": BoxPyramidStairsTerrainCfg(
                proportion=proportions["stairs_up"],
                step_height_range=(
                    terrain.stair_step_height_base,
                    terrain.stair_step_height_base + terrain.stair_step_height_gain,
                ),
                step_width=terrain.stair_step_width,
                platform_width=terrain.platform_width,
            ),
            "stairs_down": BoxInvertedPyramidStairsTerrainCfg(
                proportion=proportions["stairs_down"],
                step_height_range=(
                    terrain.stair_step_height_base,
                    terrain.stair_step_height_base + terrain.stair_step_height_gain,
                ),
                step_width=terrain.stair_step_width,
                platform_width=terrain.platform_width,
            ),
        },
    )
