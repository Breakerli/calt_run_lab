#!/usr/bin/env python3
"""Simplify ultra_GameYaw STL meshes for MuJoCo / sim2sim.

Uses Quadric Edge Collapse via ``fast_simplification`` (same strategy as
ultra88p ``meshes_simplified/``: target 50k faces, skip if already below).

Input:  legged_lab/assets/ultra_GameYaw/meshes/*.STL
Output: legged_lab/assets/ultra_GameYaw/meshes_simplified/*.STL

Requires: trimesh, fast-simplification
  pip install fast-simplification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from fast_simplification import simplify

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "legged_lab/assets/ultra_GameYaw"
MESH_IN = ASSET_DIR / "meshes"
MESH_OUT = ASSET_DIR / "meshes_simplified"
DEFAULT_TARGET = 50_000


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    n_faces = len(mesh.faces)
    if n_faces <= target_faces:
        return mesh.copy()
    vertices, faces = simplify(mesh.vertices, mesh.faces, target_count=target_faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-faces",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Max faces per mesh (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing meshes_simplified/",
    )
    args = parser.parse_args()

    if not MESH_IN.is_dir():
        raise SystemExit(f"Input directory not found: {MESH_IN}")

    if MESH_OUT.exists() and any(MESH_OUT.glob("*.STL")) and not args.force:
        raise SystemExit(
            f"{MESH_OUT} already contains STL files. Use --force to regenerate."
        )

    MESH_OUT.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    for src in sorted(MESH_IN.glob("*.STL")):
        mesh = trimesh.load(str(src), force="mesh")
        n_in = len(mesh.faces)
        out_mesh = simplify_mesh(mesh, args.target_faces)
        n_out = len(out_mesh.faces)
        dst = MESH_OUT / src.name
        out_mesh.export(str(dst))
        total_in += n_in
        total_out += n_out
        action = "copy" if n_out == n_in else "simplify"
        print(f"{src.name:35s} {n_in:8d} -> {n_out:8d} ({action})")

    print(f"{'TOTAL':35s} {total_in:8d} -> {total_out:8d}")
    print(f"Saved to {MESH_OUT}")


if __name__ == "__main__":
    main()
