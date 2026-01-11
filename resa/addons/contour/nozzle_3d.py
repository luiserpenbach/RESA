"""
3D Nozzle Geometry Generator
============================
Generates 3D geometry for rocket engine nozzles:
- Inner contour (flow path)
- Outer contour (with wall thickness)
- Surface meshes for STL export and visualization

Uses Rao-style bell nozzle contours with Bezier curve interpolation.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np

from .export import export_stl_binary, export_geometry_json


@dataclass
class NozzleParameters:
    """Input parameters for nozzle design."""
    R_t: float              # Throat radius [m]
    CR: float               # Contraction ratio (A_c / A_t)
    L_star: float           # Characteristic length [m]
    ER: float               # Expansion ratio (A_e / A_t)
    theta_n: float = 30.0   # Initial parabola angle [degrees]
    theta_e: float = 8.0    # Exit angle [degrees]
    L_percent: float = 80.0 # Bell length as % of 15-degree conical nozzle


@dataclass
class NozzleGeometry:
    """Computed nozzle geometry dimensions."""
    R_c: float      # Chamber radius [m]
    R_e: float      # Exit radius [m]
    L_c: float      # Chamber length [m]
    L_conv: float   # Convergent section length [m]
    L_div: float    # Divergent section length [m]
    L_total: float  # Total nozzle length [m]
    V_c: float      # Chamber volume [m^3]


def compute_nozzle_geometry(params: NozzleParameters) -> NozzleGeometry:
    """
    Compute derived nozzle geometry from input parameters.

    Parameters
    ----------
    params : NozzleParameters
        Input design parameters

    Returns
    -------
    NozzleGeometry
        Computed geometry dimensions
    """
    R_t = params.R_t
    CR = params.CR
    L_star = params.L_star
    ER = params.ER
    L_percent = params.L_percent

    A_t = np.pi * R_t ** 2
    R_c = R_t * np.sqrt(CR)
    R_e = R_t * np.sqrt(ER)
    V_c = L_star * A_t

    # Standard throat radii of curvature
    R_up = 1.5 * R_t
    R_down = 0.382 * R_t
    theta_conv = np.radians(35)

    # Convergent section geometry
    dx_up = R_up * np.sin(theta_conv)
    dx_down = R_down * np.sin(theta_conv)
    dy_cone = max(
        (R_c - R_up * (1 - np.cos(theta_conv))) -
        (R_t + R_down * (1 - np.cos(theta_conv))),
        0
    )
    dx_cone = dy_cone / np.tan(theta_conv) if dy_cone > 0 else 0
    L_conv = dx_up + dx_cone + dx_down

    # Chamber length from volume requirement
    V_conv_approx = (np.pi / 3) * L_conv * (R_c ** 2 + R_c * R_t + R_t ** 2)
    V_cyl = V_c - V_conv_approx * 0.7
    L_c = max(V_cyl / (np.pi * R_c ** 2), 2 * R_c)

    # Divergent section length (percent of 15-degree cone)
    L_15_cone = (R_e - R_t) / np.tan(np.radians(15))
    L_div = (L_percent / 100) * L_15_cone
    L_total = L_c + L_conv + L_div

    return NozzleGeometry(
        R_c=R_c, R_e=R_e, L_c=L_c, L_conv=L_conv,
        L_div=L_div, L_total=L_total, V_c=V_c
    )


def _cubic_bezier(P0: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                  P3: np.ndarray, t: float) -> np.ndarray:
    """Compute cubic Bezier curve point at parameter t."""
    mt = 1 - t
    return (mt ** 3 * P0 + 3 * mt ** 2 * t * P1 +
            3 * mt * t ** 2 * P2 + t ** 3 * P3)


def generate_2d_contour(
    params: NozzleParameters,
    n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D nozzle inner contour (axial position vs radius).

    Parameters
    ----------
    params : NozzleParameters
        Nozzle design parameters
    n_points : int, optional
        Number of contour points

    Returns
    -------
    x : np.ndarray
        Axial coordinates [m]
    r : np.ndarray
        Radial coordinates [m]
    """
    geom = compute_nozzle_geometry(params)
    R_t = params.R_t
    R_c = geom.R_c
    R_e = geom.R_e
    L_c = geom.L_c
    L_conv = geom.L_conv
    L_div = geom.L_div

    theta_n_rad = np.radians(params.theta_n)
    theta_e_rad = np.radians(params.theta_e)
    theta_conv = np.radians(35)
    R_up = 1.5 * R_t
    R_down = 0.382 * R_t

    x_coords = []
    r_coords = []

    # Chamber (cylindrical section)
    n_chamber = n_points // 5
    x_chamber = np.linspace(0, L_c, n_chamber)
    r_chamber = np.full_like(x_chamber, R_c)
    x_coords.extend(x_chamber.tolist())
    r_coords.extend(r_chamber.tolist())

    # Convergent - upstream arc
    n_conv_up = n_points // 6
    phi_up = np.linspace(np.pi / 2, np.pi / 2 - theta_conv, n_conv_up)
    x_up_center = L_c
    y_up_center = R_c - R_up
    x_up = x_up_center + R_up * np.cos(phi_up)
    r_up = y_up_center + R_up * np.sin(phi_up)
    x_coords.extend(x_up[1:].tolist())
    r_coords.extend(r_up[1:].tolist())

    # Convergent - conical section
    x_throat = L_c + L_conv

    # Find where downstream arc starts
    phi_test = np.linspace(0, theta_conv, 50)
    x_down_test = x_throat - R_down * np.sin(phi_test)
    r_down_test = R_t + R_down * (1 - np.cos(phi_test))
    x_cone_end = x_down_test[-1]
    r_cone_end = r_down_test[-1]

    n_cone = n_points // 8
    x_cone = np.linspace(x_up[-1], x_cone_end, n_cone)
    r_cone = np.linspace(r_up[-1], r_cone_end, n_cone)
    x_coords.extend(x_cone[1:].tolist())
    r_coords.extend(r_cone[1:].tolist())

    # Convergent - downstream arc to throat
    n_conv_down = n_points // 6
    phi_down = np.linspace(theta_conv, 0, n_conv_down)
    x_down = x_throat - R_down * np.sin(phi_down)
    r_down = R_t + R_down * (1 - np.cos(phi_down))
    x_coords.extend(x_down[1:].tolist())
    r_coords.extend(r_down[1:].tolist())

    # Bell nozzle - throat downstream arc
    n_bell_arc = n_points // 8
    phi_arc = np.linspace(0, theta_n_rad, n_bell_arc)
    x_arc = x_throat + R_down * np.sin(phi_arc)
    r_arc = R_t + R_down * (1 - np.cos(phi_arc))
    x_coords.extend(x_arc[1:].tolist())
    r_coords.extend(r_arc[1:].tolist())

    # Bell nozzle - Bezier curve to exit
    x_N = x_arc[-1]
    r_N = r_arc[-1]
    x_E = x_throat + L_div
    r_E = R_e

    P0 = np.array([x_N, r_N])
    P3 = np.array([x_E, r_E])
    L_bezier = x_E - x_N
    d1 = 0.35 * L_bezier
    d2 = 0.35 * L_bezier
    P1 = P0 + d1 * np.array([np.cos(theta_n_rad), np.sin(theta_n_rad)])
    P2 = P3 - d2 * np.array([np.cos(theta_e_rad), np.sin(theta_e_rad)])

    n_bezier = n_points // 3
    t_vals = np.linspace(0, 1, n_bezier)
    for t in t_vals[1:]:
        pt = _cubic_bezier(P0, P1, P2, P3, t)
        x_coords.append(pt[0])
        r_coords.append(pt[1])

    return np.array(x_coords), np.array(r_coords)


def generate_3d_surface_revolution(
    x_profile: np.ndarray,
    r_profile: np.ndarray,
    n_theta: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D surface by revolving 2D profile around X-axis.

    Parameters
    ----------
    x_profile : np.ndarray
        Axial coordinates
    r_profile : np.ndarray
        Radial coordinates
    n_theta : int, optional
        Number of circumferential divisions

    Returns
    -------
    X, Y, Z : np.ndarray
        3D coordinate arrays for the surface mesh
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)

    X = np.outer(x_profile, np.ones(n_theta))
    R = np.outer(r_profile, np.ones(n_theta))
    THETA = np.outer(np.ones(len(x_profile)), theta)

    Y = R * np.cos(THETA)
    Z = R * np.sin(THETA)

    return X, Y, Z


def generate_nozzle_mesh(
    params: NozzleParameters,
    n_axial: int = 100,
    n_theta: int = 64
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate nozzle inner surface mesh (vertices and faces).

    Parameters
    ----------
    params : NozzleParameters
        Nozzle design parameters
    n_axial : int, optional
        Number of axial points
    n_theta : int, optional
        Number of circumferential points

    Returns
    -------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3)
    faces : List[List[int]]
        Quad face indices
    """
    x_profile, r_profile = generate_2d_contour(params, n_axial)

    vertices = []
    faces = []

    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    for i, (x, r) in enumerate(zip(x_profile, r_profile)):
        for theta in theta_vals:
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

        if i > 0:
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                curr = i * n_theta
                prev = (i - 1) * n_theta

                faces.append([
                    prev + j,
                    curr + j,
                    curr + j_next,
                    prev + j_next
                ])

    return np.array(vertices), faces


def generate_outer_wall_mesh(
    params: NozzleParameters,
    wall_thickness: float,
    n_axial: int = 100,
    n_theta: int = 64
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate outer wall surface mesh.

    Parameters
    ----------
    params : NozzleParameters
        Nozzle design parameters
    wall_thickness : float
        Total wall thickness [m]
    n_axial : int, optional
        Number of axial points
    n_theta : int, optional
        Number of circumferential points

    Returns
    -------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3)
    faces : List[List[int]]
        Quad face indices
    """
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    r_outer = r_profile + wall_thickness

    vertices = []
    faces = []

    theta_vals = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    for i, (x, r) in enumerate(zip(x_profile, r_outer)):
        for theta in theta_vals:
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

        if i > 0:
            for j in range(n_theta):
                j_next = (j + 1) % n_theta
                curr = i * n_theta
                prev = (i - 1) * n_theta

                # Reverse winding for outer surface (normals point outward)
                faces.append([
                    prev + j,
                    prev + j_next,
                    curr + j_next,
                    curr + j
                ])

    return np.array(vertices), faces


class Nozzle3DGenerator:
    """
    3D Nozzle geometry generator with mesh and STL export capabilities.

    Parameters
    ----------
    params : NozzleParameters
        Nozzle design parameters

    Attributes
    ----------
    params : NozzleParameters
        Design parameters
    geometry : NozzleGeometry
        Computed geometry dimensions
    """

    def __init__(self, params: NozzleParameters):
        self.params = params
        self.geometry = compute_nozzle_geometry(params)

    def get_2d_contour(self, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2D nozzle contour."""
        return generate_2d_contour(self.params, n_points)

    def get_inner_mesh(
        self,
        n_axial: int = 100,
        n_theta: int = 64
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Generate inner surface mesh."""
        return generate_nozzle_mesh(self.params, n_axial, n_theta)

    def get_outer_mesh(
        self,
        wall_thickness: float,
        n_axial: int = 100,
        n_theta: int = 64
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Generate outer surface mesh."""
        return generate_outer_wall_mesh(
            self.params, wall_thickness, n_axial, n_theta
        )

    def export_inner_stl(
        self,
        filename: str,
        n_axial: int = 100,
        n_theta: int = 64,
        scale: float = 1000.0
    ) -> int:
        """
        Export inner nozzle surface to STL.

        Parameters
        ----------
        filename : str
            Output file path
        n_axial : int, optional
            Number of axial points
        n_theta : int, optional
            Number of circumferential points
        scale : float, optional
            Scale factor (default: m to mm)

        Returns
        -------
        int
            Number of triangles exported
        """
        vertices, faces = self.get_inner_mesh(n_axial, n_theta)
        return export_stl_binary(vertices, faces, filename, scale)

    def export_outer_stl(
        self,
        filename: str,
        wall_thickness: float,
        n_axial: int = 100,
        n_theta: int = 64,
        scale: float = 1000.0
    ) -> int:
        """
        Export outer nozzle surface to STL.

        Parameters
        ----------
        filename : str
            Output file path
        wall_thickness : float
            Total wall thickness [m]
        n_axial : int, optional
            Number of axial points
        n_theta : int, optional
            Number of circumferential points
        scale : float, optional
            Scale factor (default: m to mm)

        Returns
        -------
        int
            Number of triangles exported
        """
        vertices, faces = self.get_outer_mesh(wall_thickness, n_axial, n_theta)
        return export_stl_binary(vertices, faces, filename, scale)

    def export_geometry_data(self, filename: str) -> None:
        """
        Export geometry data to JSON.

        Parameters
        ----------
        filename : str
            Output file path
        """
        x, r = self.get_2d_contour()
        geom = self.geometry

        data = {
            'params': {
                'R_t': self.params.R_t,
                'CR': self.params.CR,
                'L_star': self.params.L_star,
                'ER': self.params.ER,
                'theta_n': self.params.theta_n,
                'theta_e': self.params.theta_e,
                'L_percent': self.params.L_percent
            },
            'geometry': {
                'R_c': geom.R_c,
                'R_e': geom.R_e,
                'L_c': geom.L_c,
                'L_conv': geom.L_conv,
                'L_div': geom.L_div,
                'L_total': geom.L_total,
                'V_c': geom.V_c
            },
            'contour': {
                'x': x.tolist(),
                'r': r.tolist()
            }
        }

        export_geometry_json(data, filename)

    def get_summary(self) -> dict:
        """Get geometry summary dictionary."""
        geom = self.geometry
        return {
            'total_length_mm': geom.L_total * 1000,
            'chamber_radius_mm': geom.R_c * 1000,
            'throat_radius_mm': self.params.R_t * 1000,
            'exit_radius_mm': geom.R_e * 1000,
            'chamber_length_mm': geom.L_c * 1000,
            'convergent_length_mm': geom.L_conv * 1000,
            'divergent_length_mm': geom.L_div * 1000,
            'chamber_volume_cc': geom.V_c * 1e6
        }
