from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import vpython as vp

_PARTS = ['parts.npz']

def load_mesh(canvas: Optional[vp.canvas] = None, color: vp.vector = vp.color.white, opacity: float = 1.0) -> Union[vp.compound, List[vp.compound]]:
    here = Path(__file__).resolve().parent
    compounds: List[vp.compound] = []
    for name in _PARTS:
        data = np.load(here / name)
        verts = data["verts"].copy()

        # rotate 180° around Y (yaw): (x, y, z) -> (-x, y, -z)
        verts[:, 0] *= -1
        verts[:, 2] *= -1
        
        normals = data['normals']
        faces = data['faces']
        vps = []
        for i in range(len(verts)):
            vps.append(vp.vertex(pos=vp.vector(float(verts[i,0]), float(verts[i,1]), float(verts[i,2])), normal=vp.vector(float(normals[i,0]), float(normals[i,1]), float(normals[i,2])), color=color, opacity=opacity))
        tris = []
        for (i0, i1, i2) in faces:
            tris.append(vp.triangle(v0=vps[int(i0)], v1=vps[int(i1)], v2=vps[int(i2)]))
        compounds.append(vp.compound(tris))
    model = compounds[0] if len(compounds) == 1 else compounds
    if canvas is not None:
        try:
            if isinstance(model, list):
                for c in model: c.canvas = canvas
            else:
                model.canvas = canvas
        except Exception:
            pass
    return model