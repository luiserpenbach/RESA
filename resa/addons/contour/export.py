"""
STL and JSON Export Functions for 3D Nozzle Geometry
=====================================================
Provides binary STL export and JSON serialization for
nozzle and cooling channel geometries.
"""

import json
import struct
from pathlib import Path
from typing import List, Dict, Any, Union

import numpy as np


def compute_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute normal vector for a triangle face."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    return normal


def triangulate_faces(faces: List[List[int]]) -> List[List[int]]:
    """
    Convert polygon faces to triangles.

    Handles triangles, quads, and arbitrary polygons using fan triangulation.
    """
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            # Split quad into two triangles
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # Fan triangulation for larger polygons
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])
    return triangles


def export_stl_binary(
    vertices: np.ndarray,
    faces: List[List[int]],
    filename: Union[str, Path],
    scale: float = 1000.0,
    header: str = "Binary STL - RESA Nozzle Generator"
) -> int:
    """
    Export mesh to binary STL format.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3)
    faces : List[List[int]]
        Face indices (can be triangles, quads, or polygons)
    filename : str or Path
        Output file path
    scale : float, optional
        Scale factor (default 1000.0 converts m to mm)
    header : str, optional
        STL header text (max 80 chars)

    Returns
    -------
    int
        Number of triangles written
    """
    vertices = np.array(vertices) * scale
    triangles = triangulate_faces(faces)

    # Ensure header is exactly 80 bytes
    header_bytes = header.encode('ascii')[:80]
    header_bytes = header_bytes + b'\0' * (80 - len(header_bytes))

    with open(filename, 'wb') as f:
        # Header (80 bytes)
        f.write(header_bytes)

        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))

        # Write triangles
        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            normal = compute_normal(v0, v1, v2)

            # Normal vector (3 floats)
            f.write(struct.pack('<fff', *normal))
            # Vertices (9 floats)
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            # Attribute byte count
            f.write(struct.pack('<H', 0))

    return len(triangles)


def export_stl_ascii(
    vertices: np.ndarray,
    faces: List[List[int]],
    filename: Union[str, Path],
    scale: float = 1000.0,
    solid_name: str = "nozzle"
) -> int:
    """
    Export mesh to ASCII STL format.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3)
    faces : List[List[int]]
        Face indices
    filename : str or Path
        Output file path
    scale : float, optional
        Scale factor (default 1000.0 converts m to mm)
    solid_name : str, optional
        Name of the solid

    Returns
    -------
    int
        Number of triangles written
    """
    vertices = np.array(vertices) * scale
    triangles = triangulate_faces(faces)

    with open(filename, 'w') as f:
        f.write(f"solid {solid_name}\n")

        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            normal = compute_normal(v0, v1, v2)

            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")

        f.write(f"endsolid {solid_name}\n")

    return len(triangles)


def export_geometry_json(
    data: Dict[str, Any],
    filename: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Export geometry data to JSON format.

    Handles numpy arrays by converting to lists.

    Parameters
    ----------
    data : dict
        Geometry data dictionary
    filename : str or Path
        Output file path
    indent : int, optional
        JSON indentation level
    """
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    converted_data = convert_numpy(data)

    with open(filename, 'w') as f:
        json.dump(converted_data, f, indent=indent)


def load_geometry_json(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Load geometry data from JSON file.

    Parameters
    ----------
    filename : str or Path
        Input file path

    Returns
    -------
    dict
        Loaded geometry data
    """
    with open(filename, 'r') as f:
        return json.load(f)


def export_mesh_obj(
    vertices: np.ndarray,
    faces: List[List[int]],
    filename: Union[str, Path],
    scale: float = 1000.0
) -> None:
    """
    Export mesh to Wavefront OBJ format.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3)
    faces : List[List[int]]
        Face indices (0-indexed, will be converted to 1-indexed for OBJ)
    filename : str or Path
        Output file path
    scale : float, optional
        Scale factor
    """
    vertices = np.array(vertices) * scale

    with open(filename, 'w') as f:
        f.write("# RESA Nozzle Generator - OBJ Export\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-indexed vertices)
        for face in faces:
            indices = ' '.join(str(i + 1) for i in face)
            f.write(f"f {indices}\n")
