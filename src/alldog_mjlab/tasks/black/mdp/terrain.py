"""Black terrain interfaces."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjlab.terrains import SubTerrainCfg
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from scipy import interpolate


@dataclass(kw_only=True)
class BlackRoughSlopeTerrainCfg(SubTerrainCfg):
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
        hfield_z_offset = elevation_min * self.vertical_scale
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
        physical_heights = hfield_z_offset + normalized_elevation * max_physical_height
        material_name = color_by_height(spec, noise, unique_id, physical_heights)
        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, hfield_z_offset],
            material=material_name,
        )
        x1 = int((self.size[0] / 2 - 1.0) / self.horizontal_scale)
        x2 = int((self.size[0] / 2 + 1.0) / self.horizontal_scale)
        y1 = int((self.size[1] / 2 - 1.0) / self.horizontal_scale)
        y2 = int((self.size[1] / 2 + 1.0) / self.horizontal_scale)
        spawn_height = float(np.max(noise[x1:x2, y1:y2])) * self.vertical_scale
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])
        if self.flat_patch_sampling is not None:
            raise NotImplementedError(
                "BlackRoughSlopeTerrainCfg 未实现 flat patch sampling"
            )
        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=None)

    def _slope_noise(self, difficulty: float) -> np.ndarray:
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
        func = interpolate.RectBivariateSpline(
            x, y, height_field_downsampled, kx=1, ky=1
        )
        x_upsampled = np.linspace(0, self.size[0], width_pixels)
        y_upsampled = np.linspace(0, self.size[1], length_pixels)
        z_upsampled = func(x_upsampled, y_upsampled)
        return np.rint(z_upsampled).astype(np.int16)
