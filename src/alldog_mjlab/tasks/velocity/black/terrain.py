"""Black rough 的 terrain 数学与 generator 组装。

文件角色划分：

    params.py     terrain 数值（patch 尺寸、难度、各 terrain 比例、noise 幅值）
    env_cfgs.py   terrain 装配（terrain_type / max_init_terrain_level / curriculum term）
    本文件        terrain 数学（task-local rough slope primitive + generator 组装）

legacy Black（super-dog `legged_gym/utils/terrain.py`）的 terrain 由 Isaac Gym
`terrain_utils` 组合而成，其中 rough slope 是

    pyramid_sloped_terrain(slope = 0.7 * difficulty, platform_size = 3.)
    + random_uniform_terrain(min = -amplitude, max = +amplitude,
                             step = 0.005, downsampled_scale = 0.2)
    amplitude = 0.015 + 0.1 * difficulty

MjLab v1.6.0 没有「slope + noise」的 native composition：每个 native heightfield
terrain 都是「算一个 int16 高度场数组 → 建 hfield + geom」。因此本文件的
`BlackRoughSlopeTerrainCfg` 复用两个 native terrain 的数学（slope 部分与
`HfPyramidSlopedTerrainCfg` 的 `border_width=0` 分支一致，noise 部分与
`HfRandomUniformTerrainCfg` 的 `border_width=0` 分支一致），把两个 int16 数组相加，
再走与 native 相同的 hfield / material / spawn origin 构造（复用 native 的
`color_by_height`）。没有新增 terrain framework、没有第二套 generator。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import interpolate

from mjlab.terrains import (
    BoxFlatTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    SubTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from alldog_mjlab.tasks.velocity.black import params


@dataclass(kw_only=True)
class BlackRoughSlopeTerrainCfg(SubTerrainCfg):
    """pyramid slope + random uniform noise（legacy Black 的 rough slope）。

    难度语义（difficulty d ∈ [0, 0.9]）：

        slope     = slope_range[0] + d * (slope_range[1] - slope_range[0])
                  = 0.7 * d
        amplitude = noise_base + noise_gain * d
                  = 0.015 + 0.1 * d

    slope 部分是四向金字塔（中心为 flat platform，向外下降），与
    `HfPyramidSlopedTerrainCfg(inverted=False)` 一致（同样是「在 platform 角点高度
    截断」的 plateau）；noise 部分是 ±amplitude 内、以 `noise_step` 量化、以
    `noise_downsample` 间距采样再双线性插值的均匀噪声，与 legacy
    `random_uniform_terrain` 一致。噪声叠加在整个 patch 上（包含 spawn platform），
    与 legacy 的 `height_field_raw += noise` 相同。
    """

    slope_range: tuple[float, float] = (0.0, 0.7)
    """Slope 梯度范围（rise / run），按 difficulty 插值。"""

    noise_base: float = 0.015
    """difficulty = 0 时的噪声幅值 [m]。"""

    noise_gain: float = 0.1
    """噪声幅值随 difficulty 的增量 [m]。"""

    noise_step: float = 0.005
    """噪声高度量化步长 [m]。"""

    noise_downsample: float = 0.2
    """噪声采样点间距 [m]（采样后双线性插值到高度场分辨率）。"""

    platform_width: float = 3.0
    """中心 flat platform 边长 [m]。"""

    horizontal_scale: float = 0.1
    """高度场水平分辨率 [m/cell]。"""

    vertical_scale: float = 0.005
    """高度场高度量化 [m/unit]。"""

    base_thickness_ratio: float = 1.0
    """高度场 base 厚度与最大表面高度之比。"""

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

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio

        if elevation_range > 0:
            normalized_elevation = (noise - elevation_min) / elevation_range
        else:
            normalized_elevation = np.zeros_like(noise)

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
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

        physical_heights = normalized_elevation * max_physical_height
        material_name = color_by_height(spec, noise, unique_id, physical_heights)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, 0.0],
            material=material_name,
        )

        # 非 inverted 金字塔的最高点是中心 platform，因此 spawn 高度取全局最大表面高度。
        spawn_height = max_physical_height
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        # Black rough v1 不使用 flat patch sampling（reset 沿用 flat contract）。
        if self.flat_patch_sampling is not None:
            raise NotImplementedError(
                "BlackRoughSlopeTerrainCfg 未实现 flat patch sampling"
            )

        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
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
        # kx=ky=1（双线性）与 legacy `interp2d(kind='linear')` 一致；MjLab native
        # preset 用默认三次样条，会在采样点之间 overshoot 超出 ±amplitude（见 MIGRATION.md）。
        func = interpolate.RectBivariateSpline(x, y, height_field_downsampled, kx=1, ky=1)

        x_upsampled = np.linspace(0, self.size[0], width_pixels)
        y_upsampled = np.linspace(0, self.size[1], length_pixels)
        z_upsampled = func(x_upsampled, y_upsampled)

        return np.rint(z_upsampled).astype(np.int16)


def black_rough_terrain_generator_cfg() -> TerrainGeneratorCfg:
    """Black rough v1 的 curriculum terrain generator。

    5 类 sub-terrain（curriculum 模式下每类一列）、10 行难度；`proportion` 在
    curriculum 模式下是 **env 分配权重**，不是列数。
    """
    terrain_params = params.terrain
    return TerrainGeneratorCfg(
        curriculum=True,
        size=terrain_params.size,
        border_width=terrain_params.border_width,
        num_rows=terrain_params.num_rows,
        difficulty_range=terrain_params.difficulty_range,
        add_lights=True,
        sub_terrains={
            "flat": BoxFlatTerrainCfg(proportion=terrain_params.flat_proportion),
            "smooth_slope_up": HfPyramidSlopedTerrainCfg(
                proportion=terrain_params.smooth_slope_up_proportion,
                slope_range=terrain_params.slope_range,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
                inverted=False,
            ),
            "smooth_slope_down": HfPyramidSlopedTerrainCfg(
                proportion=terrain_params.smooth_slope_down_proportion,
                slope_range=terrain_params.slope_range,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
                inverted=True,
            ),
            "rough_slope": BlackRoughSlopeTerrainCfg(
                proportion=terrain_params.rough_slope_proportion,
                slope_range=terrain_params.slope_range,
                noise_base=terrain_params.rough_noise_base,
                noise_gain=terrain_params.rough_noise_gain,
                noise_step=terrain_params.rough_noise_step,
                noise_downsample=terrain_params.rough_noise_downsample,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
            ),
            "discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
                proportion=terrain_params.obstacles_proportion,
                obstacle_height_mode="choice",
                obstacle_width_range=terrain_params.obstacle_width_range,
                obstacle_height_range=terrain_params.obstacle_height_range,
                num_obstacles=terrain_params.obstacle_count,
                platform_width=terrain_params.platform_width,
                horizontal_scale=terrain_params.horizontal_scale,
                vertical_scale=terrain_params.vertical_scale,
                square_obstacles=False,
            ),
        },
    )
