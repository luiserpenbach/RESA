"""
3D Cooling Channel Generator
=============================
Generates 3D geometry for regenerative cooling channels:
- Straight axial channels
- Helical (spiral) channels with variable helix angle
- Channel cross-sections and swept volumes for manufacturing

Channels are generated as positive bodies suitable for machining or 3D printing.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import numpy as np

from .nozzle_3d import (
    NozzleParameters,
    NozzleGeometry,
    compute_nozzle_geometry,
    generate_2d_contour
)
from .export import export_stl_binary, export_geometry_json


@dataclass
class CoolingChannelParameters:
    """Parameters for regenerative cooling channels."""
    # Wall thicknesses
    inner_wall_thickness: float = 0.001    # Inner wall (hot gas side) [m]
    outer_wall_thickness: float = 0.002    # Outer wall (structural) [m]

    # Channel geometry at throat (reference)
    channel_width_throat: float = 0.003    # Channel width at throat [m]
    channel_height_throat: float = 0.004   # Channel height at throat [m]
    rib_width_throat: float = 0.002        # Rib width between channels at throat [m]

    # Channel variation parameters
    channel_width_ratio_chamber: float = 1.5   # Width ratio at chamber vs throat
    channel_width_ratio_exit: float = 1.8      # Width ratio at exit vs throat

    # Number of channels (if None, computed from throat geometry)
    n_channels: Optional[int] = None

    # Fillet radius for channel corners
    fillet_radius: float = 0.0003  # [m]

    # Helical channel parameters
    channel_type: str = 'straight'    # 'straight' or 'helical'
    helix_angle_throat: float = 15.0  # Helix angle at throat [degrees]
    helix_angle_chamber: float = 10.0 # Helix angle at chamber [degrees]
    helix_angle_exit: float = 20.0    # Helix angle at exit [degrees]
    helix_direction: int = 1          # 1 = right-hand, -1 = left-hand


def compute_channel_count(
    R_t: float,
    channel_width: float,
    rib_width: float,
    min_channels: int = 12
) -> int:
    """
    Compute number of channels from throat geometry.

    Parameters
    ----------
    R_t : float
        Throat radius [m]
    channel_width : float
        Channel width [m]
    rib_width : float
        Rib width [m]
    min_channels : int, optional
        Minimum number of channels

    Returns
    -------
    int
        Number of channels
    """
    circumference = 2 * np.pi * R_t
    n_channels = int(circumference / (channel_width + rib_width))
    return max(n_channels, min_channels)


def compute_channel_geometry_at_x(
    x: float,
    r_inner: float,
    nozzle_params: NozzleParameters,
    channel_params: CoolingChannelParameters,
    geom: NozzleGeometry
) -> Dict[str, Any]:
    """
    Compute cooling channel geometry at a specific axial location.

    Parameters
    ----------
    x : float
        Axial position [m]
    r_inner : float
        Inner wall radius at this position [m]
    nozzle_params : NozzleParameters
        Nozzle parameters
    channel_params : CoolingChannelParameters
        Channel parameters
    geom : NozzleGeometry
        Computed nozzle geometry

    Returns
    -------
    dict
        Channel geometry at this location
    """
    x_throat = geom.L_c + geom.L_conv
    R_t = nozzle_params.R_t

    # Reference dimensions at throat
    w_throat = channel_params.channel_width_throat
    h_throat = channel_params.channel_height_throat
    rib_throat = channel_params.rib_width_throat

    # Compute number of channels
    if channel_params.n_channels is None:
        n_channels = compute_channel_count(R_t, w_throat, rib_throat)
    else:
        n_channels = channel_params.n_channels

    # Channel width variation along nozzle
    if x < geom.L_c:  # Chamber
        width_ratio = channel_params.channel_width_ratio_chamber
    elif x > x_throat:  # Divergent
        t = (x - x_throat) / geom.L_div if geom.L_div > 0 else 0
        width_ratio = 1.0 + t * (channel_params.channel_width_ratio_exit - 1.0)
    else:  # Convergent
        t = (x - geom.L_c) / geom.L_conv if geom.L_conv > 0 else 0
        width_ratio = channel_params.channel_width_ratio_chamber + \
                      t * (1.0 - channel_params.channel_width_ratio_chamber)

    channel_width = w_throat * width_ratio

    # Channel height (slight variation with radius)
    r_ratio = r_inner / R_t
    channel_height = h_throat * np.clip(0.8 + 0.2 * r_ratio, 0.7, 1.3)

    # Rib width to maintain n_channels
    circumference = 2 * np.pi * r_inner
    rib_width = (circumference - n_channels * channel_width) / n_channels
    rib_width = max(rib_width, rib_throat * 0.5)

    # Angular dimensions
    theta_channel = channel_width / r_inner
    theta_rib = rib_width / r_inner
    theta_pitch = theta_channel + theta_rib

    return {
        'n_channels': n_channels,
        'channel_width': channel_width,
        'channel_height': channel_height,
        'rib_width': rib_width,
        'theta_channel': theta_channel,
        'theta_rib': theta_rib,
        'theta_pitch': theta_pitch
    }


def compute_helix_angle_at_x(
    x: float,
    nozzle_params: NozzleParameters,
    channel_params: CoolingChannelParameters,
    geom: NozzleGeometry
) -> float:
    """
    Compute helix angle at a specific axial location.

    Uses smooth interpolation between chamber, throat, and exit angles.

    Parameters
    ----------
    x : float
        Axial position [m]
    nozzle_params : NozzleParameters
        Nozzle parameters
    channel_params : CoolingChannelParameters
        Channel parameters
    geom : NozzleGeometry
        Computed nozzle geometry

    Returns
    -------
    float
        Helix angle in radians
    """
    x_throat = geom.L_c + geom.L_conv

    angle_chamber = np.radians(channel_params.helix_angle_chamber)
    angle_throat = np.radians(channel_params.helix_angle_throat)
    angle_exit = np.radians(channel_params.helix_angle_exit)

    if x <= geom.L_c:  # Chamber
        return angle_chamber
    elif x <= x_throat:  # Convergent
        t = (x - geom.L_c) / geom.L_conv if geom.L_conv > 0 else 1
        t_smooth = t * t * (3 - 2 * t)  # Smoothstep
        return angle_chamber + t_smooth * (angle_throat - angle_chamber)
    else:  # Divergent
        t = (x - x_throat) / geom.L_div if geom.L_div > 0 else 1
        t = min(t, 1.0)
        t_smooth = t * t * (3 - 2 * t)
        return angle_throat + t_smooth * (angle_exit - angle_throat)


def compute_helix_path(
    x_profile: np.ndarray,
    r_profile: np.ndarray,
    channel_idx: int,
    n_channels: int,
    nozzle_params: NozzleParameters,
    channel_params: CoolingChannelParameters,
    geom: NozzleGeometry
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 3D helical path for a single cooling channel centerline.

    Parameters
    ----------
    x_profile : np.ndarray
        Axial coordinates
    r_profile : np.ndarray
        Inner wall radii
    channel_idx : int
        Channel index (0 to n_channels-1)
    n_channels : int
        Total number of channels
    nozzle_params : NozzleParameters
        Nozzle parameters
    channel_params : CoolingChannelParameters
        Channel parameters
    geom : NozzleGeometry
        Computed nozzle geometry

    Returns
    -------
    x_3d, y_3d, z_3d : np.ndarray
        3D coordinates of centerline
    theta : np.ndarray
        Angular position at each point
    """
    direction = channel_params.helix_direction
    theta_start = channel_idx * 2 * np.pi / n_channels

    # Integrate theta along path
    theta = np.zeros(len(x_profile))
    theta[0] = theta_start

    for i in range(1, len(x_profile)):
        dx = x_profile[i] - x_profile[i - 1]
        r_avg = (r_profile[i] + r_profile[i - 1]) / 2

        helix_angle = compute_helix_angle_at_x(
            x_profile[i], nozzle_params, channel_params, geom
        )

        if r_avg > 0:
            dtheta = direction * np.tan(helix_angle) * dx / r_avg
        else:
            dtheta = 0

        theta[i] = theta[i - 1] + dtheta

    # Channel centerline radius
    r_channel = (r_profile + channel_params.inner_wall_thickness +
                 channel_params.channel_height_throat / 2)

    x_3d = x_profile
    y_3d = r_channel * np.cos(theta)
    z_3d = r_channel * np.sin(theta)

    return x_3d, y_3d, z_3d, theta


def generate_straight_channel_mesh(
    nozzle_params: NozzleParameters,
    channel_params: CoolingChannelParameters,
    n_axial: int = 80,
    n_cross_section: int = 8
) -> Dict[str, Any]:
    """
    Generate 3D mesh for all straight cooling channels.

    Parameters
    ----------
    nozzle_params : NozzleParameters
        Nozzle parameters
    channel_params : CoolingChannelParameters
        Channel parameters
    n_axial : int, optional
        Number of axial points
    n_cross_section : int, optional
        Points per side of cross-section

    Returns
    -------
    dict
        Channel mesh data with vertices and faces for each channel
    """
    geom = compute_nozzle_geometry(nozzle_params)
    x_profile, r_profile = generate_2d_contour(nozzle_params, n_axial)

    # Get channel count from throat
    x_throat = geom.L_c + geom.L_conv
    throat_idx = np.argmin(np.abs(x_profile - x_throat))
    ch_geom_throat = compute_channel_geometry_at_x(
        x_throat, r_profile[throat_idx],
        nozzle_params, channel_params, geom
    )
    n_channels = ch_geom_throat['n_channels']

    all_channels = []

    for ch_idx in range(n_channels):
        channel_vertices = []
        channel_faces = []

        theta_base = ch_idx * 2 * np.pi / n_channels
        vertex_offset = 0

        for i, (x, r_inner) in enumerate(zip(x_profile, r_profile)):
            ch_geom = compute_channel_geometry_at_x(
                x, r_inner, nozzle_params, channel_params, geom
            )

            r_bottom = r_inner + channel_params.inner_wall_thickness
            r_top = r_bottom + ch_geom['channel_height']
            half_theta = ch_geom['theta_channel'] / 2

            # Generate cross-section vertices
            theta_angles = np.linspace(
                theta_base - half_theta,
                theta_base + half_theta,
                n_cross_section
            )

            # Bottom arc
            for theta in theta_angles:
                y = r_bottom * np.cos(theta)
                z = r_bottom * np.sin(theta)
                channel_vertices.append([x, y, z])

            # Top arc (reversed)
            for theta in reversed(theta_angles):
                y = r_top * np.cos(theta)
                z = r_top * np.sin(theta)
                channel_vertices.append([x, y, z])

            # Generate faces
            if i > 0:
                n_per_section = 2 * n_cross_section
                prev_offset = vertex_offset - n_per_section
                curr_offset = vertex_offset

                # Bottom faces
                for j in range(n_cross_section - 1):
                    channel_faces.append([
                        prev_offset + j,
                        curr_offset + j,
                        curr_offset + j + 1,
                        prev_offset + j + 1
                    ])

                # Top faces
                top_start = n_cross_section
                for j in range(n_cross_section - 1):
                    channel_faces.append([
                        prev_offset + top_start + j,
                        prev_offset + top_start + j + 1,
                        curr_offset + top_start + j + 1,
                        curr_offset + top_start + j
                    ])

                # Side walls
                channel_faces.append([
                    prev_offset,
                    prev_offset + 2 * n_cross_section - 1,
                    curr_offset + 2 * n_cross_section - 1,
                    curr_offset
                ])
                channel_faces.append([
                    prev_offset + n_cross_section - 1,
                    curr_offset + n_cross_section - 1,
                    curr_offset + n_cross_section,
                    prev_offset + n_cross_section
                ])

            vertex_offset += 2 * n_cross_section

        # End caps
        n_per_section = 2 * n_cross_section
        front_face = list(range(n_per_section))
        channel_faces.append(front_face)

        back_start = vertex_offset - n_per_section
        back_face = list(range(back_start, back_start + n_per_section))
        back_face.reverse()
        channel_faces.append(back_face)

        all_channels.append({
            'vertices': channel_vertices,
            'faces': channel_faces,
            'channel_index': ch_idx
        })

    return {
        'channel_type': 'straight',
        'n_channels': n_channels,
        'channels': all_channels
    }


def generate_helical_channel_mesh(
    nozzle_params: NozzleParameters,
    channel_params: CoolingChannelParameters,
    n_axial: int = 100,
    n_cross_section: int = 8
) -> Dict[str, Any]:
    """
    Generate 3D mesh for all helical cooling channels.

    Parameters
    ----------
    nozzle_params : NozzleParameters
        Nozzle parameters
    channel_params : CoolingChannelParameters
        Channel parameters (must have channel_type='helical')
    n_axial : int, optional
        Number of axial points
    n_cross_section : int, optional
        Points per side of cross-section

    Returns
    -------
    dict
        Channel mesh data with vertices, faces, and centerlines
    """
    geom = compute_nozzle_geometry(nozzle_params)
    x_profile, r_profile = generate_2d_contour(nozzle_params, n_axial)

    # Get channel count
    x_throat = geom.L_c + geom.L_conv
    throat_idx = np.argmin(np.abs(x_profile - x_throat))
    ch_geom_throat = compute_channel_geometry_at_x(
        x_throat, r_profile[throat_idx],
        nozzle_params, channel_params, geom
    )
    n_channels = ch_geom_throat['n_channels']

    all_channels = []

    for ch_idx in range(n_channels):
        # Compute helical path
        x_path, y_path, z_path, theta_path = compute_helix_path(
            x_profile, r_profile, ch_idx, n_channels,
            nozzle_params, channel_params, geom
        )

        vertices = []
        faces = []
        n_per_section = 2 * n_cross_section

        for i in range(len(x_profile)):
            x = x_profile[i]
            r = r_profile[i]
            theta_center = theta_path[i]
            helix_angle = compute_helix_angle_at_x(
                x, nozzle_params, channel_params, geom
            )

            ch_geom = compute_channel_geometry_at_x(
                x, r, nozzle_params, channel_params, geom
            )

            r_bottom = r + channel_params.inner_wall_thickness
            r_top = r_bottom + ch_geom['channel_height']
            half_width_angle = ch_geom['theta_channel'] / 2

            # Adjust for helix angle
            cos_helix = np.cos(helix_angle)
            if cos_helix > 0.1:
                half_width_angle_eff = half_width_angle / cos_helix
            else:
                half_width_angle_eff = half_width_angle

            theta_angles = np.linspace(
                theta_center - half_width_angle_eff,
                theta_center + half_width_angle_eff,
                n_cross_section
            )

            # Bottom arc
            for theta in theta_angles:
                y = r_bottom * np.cos(theta)
                z = r_bottom * np.sin(theta)
                vertices.append([x, y, z])

            # Top arc (reversed)
            for theta in reversed(theta_angles):
                y = r_top * np.cos(theta)
                z = r_top * np.sin(theta)
                vertices.append([x, y, z])

            # Generate faces
            if i > 0:
                curr_offset = i * n_per_section
                prev_offset = (i - 1) * n_per_section

                # Bottom surface
                for j in range(n_cross_section - 1):
                    faces.append([
                        prev_offset + j,
                        curr_offset + j,
                        curr_offset + j + 1,
                        prev_offset + j + 1
                    ])

                # Top surface
                top_start = n_cross_section
                for j in range(n_cross_section - 1):
                    faces.append([
                        prev_offset + top_start + j,
                        prev_offset + top_start + j + 1,
                        curr_offset + top_start + j + 1,
                        curr_offset + top_start + j
                    ])

                # Side walls
                faces.append([
                    prev_offset,
                    prev_offset + n_per_section - 1,
                    curr_offset + n_per_section - 1,
                    curr_offset
                ])
                faces.append([
                    prev_offset + n_cross_section - 1,
                    curr_offset + n_cross_section - 1,
                    curr_offset + n_cross_section,
                    prev_offset + n_cross_section
                ])

        # End caps
        front_face = list(range(n_per_section))
        faces.append(front_face)

        back_start = (len(x_profile) - 1) * n_per_section
        back_face = list(range(back_start, back_start + n_per_section))
        back_face.reverse()
        faces.append(back_face)

        all_channels.append({
            'vertices': vertices,
            'faces': faces,
            'channel_index': ch_idx,
            'centerline': {
                'x': x_path.tolist(),
                'y': y_path.tolist(),
                'z': z_path.tolist(),
                'theta': theta_path.tolist()
            }
        })

    return {
        'channel_type': 'helical',
        'n_channels': n_channels,
        'helix_params': {
            'angle_chamber': channel_params.helix_angle_chamber,
            'angle_throat': channel_params.helix_angle_throat,
            'angle_exit': channel_params.helix_angle_exit,
            'direction': channel_params.helix_direction
        },
        'channels': all_channels
    }


class CoolingChannel3DGenerator:
    """
    3D Cooling channel generator with mesh and export capabilities.

    Parameters
    ----------
    nozzle_params : NozzleParameters
        Nozzle design parameters
    channel_params : CoolingChannelParameters
        Cooling channel parameters
    """

    def __init__(
        self,
        nozzle_params: NozzleParameters,
        channel_params: CoolingChannelParameters
    ):
        self.nozzle_params = nozzle_params
        self.channel_params = channel_params
        self.geometry = compute_nozzle_geometry(nozzle_params)
        self._mesh_cache: Optional[Dict[str, Any]] = None

    def generate_mesh(
        self,
        n_axial: int = 80,
        n_cross_section: int = 8
    ) -> Dict[str, Any]:
        """
        Generate channel mesh based on channel type.

        Returns
        -------
        dict
            Mesh data with vertices and faces for all channels
        """
        if self.channel_params.channel_type == 'helical':
            mesh = generate_helical_channel_mesh(
                self.nozzle_params, self.channel_params,
                n_axial, n_cross_section
            )
        else:
            mesh = generate_straight_channel_mesh(
                self.nozzle_params, self.channel_params,
                n_axial, n_cross_section
            )

        self._mesh_cache = mesh
        return mesh

    def get_channel_profiles(self, n_axial: int = 100) -> Dict[str, Any]:
        """
        Get channel geometry profiles along the nozzle.

        Returns
        -------
        dict
            Channel profiles at each axial station
        """
        geom = self.geometry
        x_profile, r_profile = generate_2d_contour(self.nozzle_params, n_axial)

        profiles = []
        for x, r in zip(x_profile, r_profile):
            ch_geom = compute_channel_geometry_at_x(
                x, r, self.nozzle_params, self.channel_params, geom
            )
            profiles.append({
                'x': float(x),
                'r_inner': float(r),
                'r_channel_bottom': float(r + self.channel_params.inner_wall_thickness),
                'r_channel_top': float(r + self.channel_params.inner_wall_thickness +
                                       ch_geom['channel_height']),
                **{k: float(v) if isinstance(v, (int, float, np.floating))
                   else v for k, v in ch_geom.items()}
            })

        return {
            'nozzle_params': {
                'R_t': self.nozzle_params.R_t,
                'CR': self.nozzle_params.CR,
                'ER': self.nozzle_params.ER,
            },
            'channel_params': {
                'inner_wall_thickness': self.channel_params.inner_wall_thickness,
                'outer_wall_thickness': self.channel_params.outer_wall_thickness,
                'channel_width_throat': self.channel_params.channel_width_throat,
                'channel_height_throat': self.channel_params.channel_height_throat,
                'channel_type': self.channel_params.channel_type,
            },
            'profiles': profiles
        }

    def export_channel_stl(
        self,
        filename: str,
        channel_index: int = 0,
        n_axial: int = 80,
        n_cross_section: int = 8,
        scale: float = 1000.0
    ) -> int:
        """
        Export a single channel to STL.

        Parameters
        ----------
        filename : str
            Output file path
        channel_index : int, optional
            Index of channel to export
        n_axial : int, optional
            Number of axial points
        n_cross_section : int, optional
            Cross-section resolution
        scale : float, optional
            Scale factor

        Returns
        -------
        int
            Number of triangles exported
        """
        if self._mesh_cache is None:
            self.generate_mesh(n_axial, n_cross_section)

        channel = self._mesh_cache['channels'][channel_index]
        return export_stl_binary(
            np.array(channel['vertices']),
            channel['faces'],
            filename,
            scale
        )

    def export_all_channels_stl(
        self,
        filename_pattern: str,
        n_axial: int = 80,
        n_cross_section: int = 8,
        scale: float = 1000.0
    ) -> int:
        """
        Export all channels to separate STL files.

        Parameters
        ----------
        filename_pattern : str
            Pattern with {} for channel index (e.g., "channel_{}.stl")
        n_axial : int, optional
            Number of axial points
        n_cross_section : int, optional
            Cross-section resolution
        scale : float, optional
            Scale factor

        Returns
        -------
        int
            Total number of triangles exported
        """
        if self._mesh_cache is None:
            self.generate_mesh(n_axial, n_cross_section)

        total_triangles = 0
        for i, channel in enumerate(self._mesh_cache['channels']):
            filename = filename_pattern.format(i)
            triangles = export_stl_binary(
                np.array(channel['vertices']),
                channel['faces'],
                filename,
                scale
            )
            total_triangles += triangles

        return total_triangles

    def export_geometry_data(self, filename: str) -> None:
        """Export channel geometry data to JSON."""
        profiles = self.get_channel_profiles()
        export_geometry_json(profiles, filename)

    def get_summary(self) -> Dict[str, Any]:
        """Get channel configuration summary."""
        x_throat = self.geometry.L_c + self.geometry.L_conv
        x_profile, r_profile = generate_2d_contour(self.nozzle_params, 100)
        throat_idx = np.argmin(np.abs(x_profile - x_throat))

        ch_geom = compute_channel_geometry_at_x(
            x_throat, r_profile[throat_idx],
            self.nozzle_params, self.channel_params, self.geometry
        )

        summary = {
            'channel_type': self.channel_params.channel_type,
            'n_channels': ch_geom['n_channels'],
            'inner_wall_mm': self.channel_params.inner_wall_thickness * 1000,
            'outer_wall_mm': self.channel_params.outer_wall_thickness * 1000,
            'channel_width_throat_mm': self.channel_params.channel_width_throat * 1000,
            'channel_height_throat_mm': self.channel_params.channel_height_throat * 1000,
            'rib_width_throat_mm': self.channel_params.rib_width_throat * 1000,
        }

        if self.channel_params.channel_type == 'helical':
            summary.update({
                'helix_angle_chamber_deg': self.channel_params.helix_angle_chamber,
                'helix_angle_throat_deg': self.channel_params.helix_angle_throat,
                'helix_angle_exit_deg': self.channel_params.helix_angle_exit,
                'helix_direction': 'right-hand' if self.channel_params.helix_direction == 1 else 'left-hand'
            })

        return summary
