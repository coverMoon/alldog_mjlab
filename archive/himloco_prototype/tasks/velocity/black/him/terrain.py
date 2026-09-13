"""Source terrain families represented as MuJoCo heightfields.

Heightfield contacts differ from Isaac Gym's slope-corrected triangle mesh.
The 20 source columns are retained (weights in the source sum to 1.1).
"""
from dataclasses import dataclass
import uuid
import mujoco
import numpy as np
from scipy.ndimage import zoom
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class BlackTerrainCfg(SubTerrainCfg):
    choice: float = 0.0

    def function(self, difficulty, spec, rng):
        nx, ny = (int(s / .1) for s in self.size)
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        pyramid = (nx/2-np.abs(nx/2-xx))/(nx/2) * (ny/2-np.abs(ny/2-yy))/(ny/2)
        z = np.zeros((nx, ny))
        c = self.choice
        if .1 <= c < .3:
            slope = .7 * difficulty * (-1 if c < .15 else 1)
            z = slope * self.size[0]/2 * pyramid
            platform_z = z[nx//2-15, ny//2-15]
            z = np.clip(z, min(0, platform_z), max(0, platform_z))
            if c >= .2:
                amplitude = .015 + .1 * difficulty
                noise = rng.uniform(-amplitude, amplitude, (nx//2, ny//2))
                z += zoom(noise, (nx/noise.shape[0], ny/noise.shape[1]), order=1)
        elif .3 <= c < .8:
            step_height = (.05 + .18*difficulty) * (-1 if c < .55 else 1)
            # Source currently uses fixed 0.3 m steps despite its width-course comment.
            depth = np.minimum.reduce([xx, yy, nx-1-xx, ny-1-yy]) * .1
            nsteps = max(0, int((min(self.size)-3.) / .6))
            z = np.minimum(np.floor(depth/.3), nsteps) * step_height
        elif c >= .8:
            height = .06 + .2*difficulty
            for _ in range(20):
                w, h = rng.integers(10, 20, size=2)
                x, y = rng.integers(0, nx-w), rng.integers(0, ny-h)
                z[x:x+w, y:y+h] = rng.choice([-height, -height/2, height/2, height])
            z[nx//2-15:nx//2+15, ny//2-15:ny//2+15] = 0
        z = np.round(z/.005)*.005
        low, high = float(z.min()), float(z.max())
        span = max(high-low, .001)
        field = spec.add_hfield(name="black_"+uuid.uuid4().hex, size=[self.size[0]/2, self.size[1]/2, span, max(span, .1)], nrow=ny, ncol=nx, userdata=((z.T-low)/span).astype(np.float32).flatten().tolist())
        geom = spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name, pos=[self.size[0]/2, self.size[1]/2, low], friction=[1., .005, .0001])
        origin = np.array([self.size[0]/2, self.size[1]/2, z[nx//2-10:nx//2+10, ny//2-10:ny//2+10].max()])
        return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)])


def terrain_cfg():
    return TerrainGeneratorCfg(size=(8., 8.), border_width=25., num_rows=10, num_cols=20, curriculum=True, difficulty_range=(0., .9), color_scheme="none", sub_terrains={f"column_{i:02}": BlackTerrainCfg(choice=i/20+.001, proportion=1.) for i in range(20)})
