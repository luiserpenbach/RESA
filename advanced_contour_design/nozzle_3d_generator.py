"""
3D Rocket Engine Nozzle & Cooling Channel Generator
====================================================
Generates 3D geometry for:
1. Nozzle inner contour (flow path)
2. Nozzle outer contour (with wall thickness)
3. Regenerative cooling channels (positive body for machining/printing)

Exports to STL format for CAD/3D printing and provides interactive visualization.
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import struct


@dataclass
class NozzleParameters:
    """Input parameters for nozzle design."""
    R_t: float          # Throat radius [m]
    CR: float           # Contraction ratio (A_c / A_t)
    L_star: float       # Characteristic length [m]
    ER: float           # Expansion ratio (A_e / A_t)
    theta_n: float = 30 # Initial parabola angle [degrees]
    theta_e: float = 8  # Exit angle [degrees]
    L_percent: float = 80  # Bell length as % of 15° conical nozzle


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
    
    # Fillet radius for channel corners (for manufacturing)
    fillet_radius: float = 0.0003  # [m]
    
    # Helical channel parameters
    channel_type: str = 'straight'  # 'straight' or 'helical'
    helix_angle_throat: float = 15.0  # Helix angle at throat [degrees] (0 = axial, 90 = circumferential)
    helix_angle_chamber: float = 10.0  # Helix angle at chamber [degrees]
    helix_angle_exit: float = 20.0     # Helix angle at exit [degrees]
    helix_direction: int = 1           # 1 = right-hand helix, -1 = left-hand helix
    
    # Bifurcation (channel splitting) - for maintaining channel density
    enable_bifurcation: bool = False
    bifurcation_ratio: float = 1.5  # Split channel when width exceeds this ratio of throat width


@dataclass 
class NozzleGeometry:
    """Computed nozzle geometry."""
    R_c: float
    R_e: float
    L_c: float
    L_conv: float
    L_div: float
    L_total: float
    V_c: float


def compute_nozzle_geometry(params: NozzleParameters) -> NozzleGeometry:
    """Compute derived nozzle geometry from input parameters."""
    R_t = params.R_t
    CR = params.CR
    L_star = params.L_star
    ER = params.ER
    L_percent = params.L_percent
    
    A_t = np.pi * R_t**2
    R_c = R_t * np.sqrt(CR)
    R_e = R_t * np.sqrt(ER)
    V_c = L_star * A_t
    
    R_up = 1.5 * R_t
    R_down = 0.382 * R_t
    theta_conv = np.radians(35)
    
    dx_up = R_up * np.sin(theta_conv)
    dx_down = R_down * np.sin(theta_conv)
    dy_cone = max((R_c - R_up * (1 - np.cos(theta_conv))) - 
                  (R_t + R_down * (1 - np.cos(theta_conv))), 0)
    dx_cone = dy_cone / np.tan(theta_conv) if dy_cone > 0 else 0
    L_conv = dx_up + dx_cone + dx_down
    
    V_conv_approx = (np.pi / 3) * L_conv * (R_c**2 + R_c * R_t + R_t**2)
    V_cyl = V_c - V_conv_approx * 0.7
    L_c = max(V_cyl / (np.pi * R_c**2), 2 * R_c)
    
    L_15_cone = (R_e - R_t) / np.tan(np.radians(15))
    L_div = (L_percent / 100) * L_15_cone
    L_total = L_c + L_conv + L_div
    
    return NozzleGeometry(R_c=R_c, R_e=R_e, L_c=L_c, L_conv=L_conv, 
                          L_div=L_div, L_total=L_total, V_c=V_c)


def cubic_bezier(P0, P1, P2, P3, t):
    """Compute cubic Bézier curve point."""
    mt = 1 - t
    return mt**3 * P0 + 3*mt**2*t * P1 + 3*mt*t**2 * P2 + t**3 * P3


def generate_2d_contour(params: NozzleParameters, n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D nozzle inner contour (axial position vs radius).
    Returns arrays of (x, r) coordinates.
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
    
    # Chamber (cylindrical)
    n_chamber = n_points // 5
    x_chamber = np.linspace(0, L_c, n_chamber)
    r_chamber = np.full_like(x_chamber, R_c)
    x_coords.extend(x_chamber.tolist())
    r_coords.extend(r_chamber.tolist())
    
    # Convergent - upstream arc
    n_conv_up = n_points // 6
    phi_up = np.linspace(np.pi/2, np.pi/2 - theta_conv, n_conv_up)
    x_up_center = L_c
    y_up_center = R_c - R_up
    x_up = x_up_center + R_up * np.cos(phi_up)
    r_up = y_up_center + R_up * np.sin(phi_up)
    x_coords.extend(x_up[1:].tolist())
    r_coords.extend(r_up[1:].tolist())
    
    # Convergent - conical section
    x_throat = L_c + L_conv
    x_cone_start = x_up[-1]
    r_cone_start = r_up[-1]
    
    # Find where downstream arc starts
    phi_test = np.linspace(0, theta_conv, 50)
    x_down_test = x_throat - R_down * np.sin(phi_test)
    r_down_test = R_t + R_down * (1 - np.cos(phi_test))
    x_cone_end = x_down_test[-1]
    r_cone_end = r_down_test[-1]
    
    n_cone = n_points // 8
    x_cone = np.linspace(x_cone_start, x_cone_end, n_cone)
    r_cone = np.linspace(r_cone_start, r_cone_end, n_cone)
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
    
    # Bell nozzle - Bézier curve
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
        pt = cubic_bezier(P0, P1, P2, P3, t)
        x_coords.append(pt[0])
        r_coords.append(pt[1])
    
    return np.array(x_coords), np.array(r_coords)


def generate_3d_surface_revolution(x_profile: np.ndarray, r_profile: np.ndarray, 
                                    n_theta: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D surface by revolving 2D profile around X-axis.
    Returns X, Y, Z coordinate arrays for the surface mesh.
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    
    # Create meshgrid
    X = np.outer(x_profile, np.ones(n_theta))
    R = np.outer(r_profile, np.ones(n_theta))
    THETA = np.outer(np.ones(len(x_profile)), theta)
    
    Y = R * np.cos(THETA)
    Z = R * np.sin(THETA)
    
    return X, Y, Z


def compute_outer_wall_radius(x: np.ndarray, r_inner: np.ndarray,
                               params: NozzleParameters,
                               cooling_params: CoolingChannelParameters,
                               geom: NozzleGeometry) -> np.ndarray:
    """
    Compute outer wall radius including inner wall + channel + outer wall.
    Channel height varies along the nozzle.
    """
    t_inner = cooling_params.inner_wall_thickness
    t_outer = cooling_params.outer_wall_thickness
    h_throat = cooling_params.channel_height_throat
    
    # Channel height variation (simplified: proportional to local heat flux needs)
    # Throat has highest heat flux, so we keep channel height relatively constant
    # but could vary it based on thermal analysis
    
    x_throat = geom.L_c + geom.L_conv
    
    # Simple variation: channel height scales slightly with radius
    r_throat = params.R_t
    height_ratio = 0.8 + 0.4 * (r_inner / r_throat)  # 0.8 to 1.2 variation
    channel_height = h_throat * np.clip(height_ratio, 0.7, 1.5)
    
    r_outer = r_inner + t_inner + channel_height + t_outer
    
    return r_outer


def compute_channel_geometry_at_x(x: float, r_inner: float,
                                   params: NozzleParameters,
                                   cooling_params: CoolingChannelParameters,
                                   geom: NozzleGeometry) -> dict:
    """
    Compute cooling channel geometry at a specific axial location.
    Returns channel width, height, and angular spacing.
    """
    x_throat = geom.L_c + geom.L_conv
    R_t = params.R_t
    
    # Reference dimensions at throat
    w_throat = cooling_params.channel_width_throat
    h_throat = cooling_params.channel_height_throat
    rib_throat = cooling_params.rib_width_throat
    
    # Compute number of channels if not specified
    if cooling_params.n_channels is None:
        circumference_throat = 2 * np.pi * R_t
        n_channels = int(circumference_throat / (w_throat + rib_throat))
        n_channels = max(n_channels, 12)  # Minimum 12 channels
    else:
        n_channels = cooling_params.n_channels
    
    # Channel width variation along nozzle
    if x < geom.L_c:  # Chamber
        width_ratio = cooling_params.channel_width_ratio_chamber
    elif x > x_throat:  # Divergent
        # Linear interpolation from throat to exit
        t = (x - x_throat) / geom.L_div if geom.L_div > 0 else 0
        width_ratio = 1.0 + t * (cooling_params.channel_width_ratio_exit - 1.0)
    else:  # Convergent
        t = (x - geom.L_c) / geom.L_conv if geom.L_conv > 0 else 0
        width_ratio = cooling_params.channel_width_ratio_chamber + t * (1.0 - cooling_params.channel_width_ratio_chamber)
    
    channel_width = w_throat * width_ratio
    
    # Channel height (relatively constant, slight variation)
    r_ratio = r_inner / R_t
    channel_height = h_throat * np.clip(0.8 + 0.2 * r_ratio, 0.7, 1.3)
    
    # Compute rib width to maintain n_channels
    circumference = 2 * np.pi * r_inner
    rib_width = (circumference - n_channels * channel_width) / n_channels
    rib_width = max(rib_width, cooling_params.rib_width_throat * 0.5)  # Minimum rib width
    
    # Angular width of channel and rib
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


def compute_helix_angle_at_x(x: float, params: NozzleParameters,
                              cooling_params: CoolingChannelParameters,
                              geom: NozzleGeometry) -> float:
    """
    Compute the helix angle at a specific axial location.
    Helix angle varies smoothly from chamber to throat to exit.
    
    Returns helix angle in radians.
    """
    x_throat = geom.L_c + geom.L_conv
    
    angle_chamber = np.radians(cooling_params.helix_angle_chamber)
    angle_throat = np.radians(cooling_params.helix_angle_throat)
    angle_exit = np.radians(cooling_params.helix_angle_exit)
    
    if x <= geom.L_c:  # Chamber region
        # Constant angle in chamber
        return angle_chamber
    elif x <= x_throat:  # Convergent section
        # Smooth transition from chamber to throat
        t = (x - geom.L_c) / geom.L_conv if geom.L_conv > 0 else 1
        # Use smoothstep for smooth transition
        t_smooth = t * t * (3 - 2 * t)
        return angle_chamber + t_smooth * (angle_throat - angle_chamber)
    else:  # Divergent section
        # Smooth transition from throat to exit
        t = (x - x_throat) / geom.L_div if geom.L_div > 0 else 1
        t = min(t, 1.0)
        t_smooth = t * t * (3 - 2 * t)
        return angle_throat + t_smooth * (angle_exit - angle_throat)


def compute_helix_path(x_profile: np.ndarray, r_profile: np.ndarray,
                        channel_idx: int, n_channels: int,
                        params: NozzleParameters,
                        cooling_params: CoolingChannelParameters,
                        geom: NozzleGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the 3D helical path for a single cooling channel centerline.
    
    The helix is defined by integrating the tangent angle along the nozzle.
    At each point, the channel makes an angle (helix_angle) with the axial direction.
    
    Returns x, y, z coordinates of the channel centerline.
    """
    direction = cooling_params.helix_direction
    
    # Starting angular position for this channel
    theta_start = channel_idx * 2 * np.pi / n_channels
    
    # Integrate theta along the path
    theta = np.zeros(len(x_profile))
    theta[0] = theta_start
    
    for i in range(1, len(x_profile)):
        dx = x_profile[i] - x_profile[i-1]
        r_avg = (r_profile[i] + r_profile[i-1]) / 2
        
        # Get helix angle at this location
        helix_angle = compute_helix_angle_at_x(x_profile[i], params, cooling_params, geom)
        
        # Angular increment: dtheta = tan(helix_angle) * dx / r
        # This comes from: tan(helix_angle) = r * dtheta / dx
        if r_avg > 0:
            dtheta = direction * np.tan(helix_angle) * dx / r_avg
        else:
            dtheta = 0
        
        theta[i] = theta[i-1] + dtheta
    
    # Compute 3D coordinates
    # The channel is on the inner wall surface + inner_wall_thickness (channel bottom)
    r_channel = r_profile + cooling_params.inner_wall_thickness + cooling_params.channel_height_throat / 2
    
    x_3d = x_profile
    y_3d = r_channel * np.cos(theta)
    z_3d = r_channel * np.sin(theta)
    
    return x_3d, y_3d, z_3d, theta


def compute_helical_channel_cross_section(x: float, r_inner: float, theta_center: float,
                                           helix_angle: float,
                                           params: NozzleParameters,
                                           cooling_params: CoolingChannelParameters,
                                           geom: NozzleGeometry,
                                           n_points: int = 8) -> List[np.ndarray]:
    """
    Compute the cross-section vertices of a helical channel at a given axial position.
    
    The cross-section is perpendicular to the helical path (not the axis).
    This accounts for the channel being tilted due to the helix angle.
    
    Returns list of 3D vertices forming the channel cross-section.
    """
    ch_geom = compute_channel_geometry_at_x(x, r_inner, params, cooling_params, geom)
    
    r_bottom = r_inner + cooling_params.inner_wall_thickness
    r_top = r_bottom + ch_geom['channel_height']
    half_width_angle = ch_geom['theta_channel'] / 2
    
    # For helical channels, the cross-section is tilted
    # The effective width in the circumferential direction increases with helix angle
    # width_effective = width / cos(helix_angle)
    cos_helix = np.cos(helix_angle)
    half_width_angle_effective = half_width_angle / cos_helix if cos_helix > 0.1 else half_width_angle
    
    vertices = []
    
    # Bottom arc (inner wall outer surface)
    theta_angles = np.linspace(theta_center - half_width_angle_effective,
                               theta_center + half_width_angle_effective, n_points)
    for theta in theta_angles:
        y = r_bottom * np.cos(theta)
        z = r_bottom * np.sin(theta)
        vertices.append(np.array([x, y, z]))
    
    # Top arc (outer wall inner surface) - reversed
    for theta in reversed(theta_angles):
        y = r_top * np.cos(theta)
        z = r_top * np.sin(theta)
        vertices.append(np.array([x, y, z]))
    
    return vertices


def generate_helical_channel_mesh(params: NozzleParameters,
                                   cooling_params: CoolingChannelParameters,
                                   n_axial: int = 100,
                                   n_cross_section: int = 8) -> dict:
    """
    Generate 3D mesh for all helical cooling channels.
    
    Returns dict with vertices and faces for each channel.
    """
    geom = compute_nozzle_geometry(params)
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    
    # Get channel count
    x_throat = geom.L_c + geom.L_conv
    throat_idx = np.argmin(np.abs(x_profile - x_throat))
    ch_geom_throat = compute_channel_geometry_at_x(x_throat, r_profile[throat_idx],
                                                    params, cooling_params, geom)
    n_channels = ch_geom_throat['n_channels']
    
    all_channels = []
    
    for ch_idx in range(n_channels):
        # Compute helical path for this channel
        x_path, y_path, z_path, theta_path = compute_helix_path(
            x_profile, r_profile, ch_idx, n_channels,
            params, cooling_params, geom
        )
        
        vertices = []
        faces = []
        
        n_per_section = 2 * n_cross_section
        
        for i in range(len(x_profile)):
            x = x_profile[i]
            r = r_profile[i]
            theta_center = theta_path[i]
            helix_angle = compute_helix_angle_at_x(x, params, cooling_params, geom)
            
            # Get cross-section vertices
            section_verts = compute_helical_channel_cross_section(
                x, r, theta_center, helix_angle,
                params, cooling_params, geom, n_cross_section
            )
            
            for v in section_verts:
                vertices.append(v.tolist())
            
            # Generate faces connecting to previous section
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
                # Left side
                faces.append([
                    prev_offset,
                    prev_offset + n_per_section - 1,
                    curr_offset + n_per_section - 1,
                    curr_offset
                ])
                # Right side
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
            'angle_chamber': cooling_params.helix_angle_chamber,
            'angle_throat': cooling_params.helix_angle_throat,
            'angle_exit': cooling_params.helix_angle_exit,
            'direction': cooling_params.helix_direction
        },
        'channels': all_channels
    }


def generate_cooling_channel_profiles(params: NozzleParameters,
                                       cooling_params: CoolingChannelParameters,
                                       n_axial: int = 100) -> dict:
    """
    Generate cooling channel geometry data along the nozzle.
    Returns dict with axial positions and channel parameters.
    """
    geom = compute_nozzle_geometry(params)
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    
    channel_data = []
    for x, r in zip(x_profile, r_profile):
        ch_geom = compute_channel_geometry_at_x(x, r, params, cooling_params, geom)
        channel_data.append({
            'x': float(x),
            'r_inner': float(r),
            'r_channel_bottom': float(r + cooling_params.inner_wall_thickness),
            'r_channel_top': float(r + cooling_params.inner_wall_thickness + ch_geom['channel_height']),
            'r_outer': float(r + cooling_params.inner_wall_thickness + ch_geom['channel_height'] + cooling_params.outer_wall_thickness),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in ch_geom.items()}
        })
    
    return {
        'params': {
            'R_t': params.R_t,
            'CR': params.CR,
            'L_star': params.L_star,
            'ER': params.ER,
            'theta_n': params.theta_n,
            'theta_e': params.theta_e,
            'L_percent': params.L_percent
        },
        'cooling_params': {
            'inner_wall_thickness': cooling_params.inner_wall_thickness,
            'outer_wall_thickness': cooling_params.outer_wall_thickness,
            'channel_width_throat': cooling_params.channel_width_throat,
            'channel_height_throat': cooling_params.channel_height_throat,
            'rib_width_throat': cooling_params.rib_width_throat
        },
        'geometry': {
            'R_c': geom.R_c,
            'R_e': geom.R_e,
            'L_c': geom.L_c,
            'L_conv': geom.L_conv,
            'L_div': geom.L_div,
            'L_total': geom.L_total
        },
        'channel_profiles': channel_data
    }


def generate_channel_3d_vertices(params: NozzleParameters,
                                  cooling_params: CoolingChannelParameters,
                                  n_axial: int = 80,
                                  n_channel_circumferential: int = 8) -> dict:
    """
    Generate 3D vertices for cooling channels (positive body for machining).
    Each channel is a 3D swept volume.
    
    Returns vertices and faces for all channels.
    """
    geom = compute_nozzle_geometry(params)
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    
    # Get channel count from throat
    x_throat = geom.L_c + geom.L_conv
    throat_idx = np.argmin(np.abs(x_profile - x_throat))
    ch_geom_throat = compute_channel_geometry_at_x(x_throat, r_profile[throat_idx], 
                                                    params, cooling_params, geom)
    n_channels = ch_geom_throat['n_channels']
    
    all_channels = []
    
    for ch_idx in range(n_channels):
        channel_vertices = []
        channel_faces = []
        
        # Angular position of this channel center
        theta_base = ch_idx * 2 * np.pi / n_channels
        
        vertex_offset = 0
        
        for i, (x, r_inner) in enumerate(zip(x_profile, r_profile)):
            ch_geom = compute_channel_geometry_at_x(x, r_inner, params, cooling_params, geom)
            
            r_bottom = r_inner + cooling_params.inner_wall_thickness
            r_top = r_bottom + ch_geom['channel_height']
            
            # Channel angular extent
            half_theta = ch_geom['theta_channel'] / 2
            
            # Generate channel cross-section vertices at this axial station
            # Rectangle cross-section (could add fillets later)
            theta_angles = np.linspace(theta_base - half_theta, 
                                       theta_base + half_theta, 
                                       n_channel_circumferential)
            
            # Bottom arc
            for theta in theta_angles:
                y = r_bottom * np.cos(theta)
                z = r_bottom * np.sin(theta)
                channel_vertices.append([x, y, z])
            
            # Top arc (reversed for proper face winding)
            for theta in reversed(theta_angles):
                y = r_top * np.cos(theta)
                z = r_top * np.sin(theta)
                channel_vertices.append([x, y, z])
            
            # Generate faces connecting to previous section
            if i > 0:
                n_per_section = 2 * n_channel_circumferential
                prev_offset = vertex_offset - n_per_section
                curr_offset = vertex_offset
                
                # Connect bottom arcs
                for j in range(n_channel_circumferential - 1):
                    channel_faces.append([
                        prev_offset + j,
                        curr_offset + j,
                        curr_offset + j + 1,
                        prev_offset + j + 1
                    ])
                
                # Connect top arcs
                top_start = n_channel_circumferential
                for j in range(n_channel_circumferential - 1):
                    channel_faces.append([
                        prev_offset + top_start + j,
                        prev_offset + top_start + j + 1,
                        curr_offset + top_start + j + 1,
                        curr_offset + top_start + j
                    ])
                
                # Connect sides (left and right walls of channel)
                # Left side
                channel_faces.append([
                    prev_offset,
                    prev_offset + 2*n_channel_circumferential - 1,
                    curr_offset + 2*n_channel_circumferential - 1,
                    curr_offset
                ])
                # Right side
                channel_faces.append([
                    prev_offset + n_channel_circumferential - 1,
                    curr_offset + n_channel_circumferential - 1,
                    curr_offset + n_channel_circumferential,
                    prev_offset + n_channel_circumferential
                ])
            
            vertex_offset += 2 * n_channel_circumferential
        
        # Cap the ends
        n_per_section = 2 * n_channel_circumferential
        
        # Front cap (x = 0)
        front_face = list(range(n_per_section))
        channel_faces.append(front_face)
        
        # Back cap (x = L_total)
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
        'n_channels': n_channels,
        'channels': all_channels
    }


def export_to_stl_binary(vertices: np.ndarray, faces: List[List[int]], 
                          filename: str, scale: float = 1000.0):
    """
    Export mesh to binary STL format.
    Scale converts from meters to mm by default.
    """
    vertices = np.array(vertices) * scale
    
    def compute_normal(v0, v1, v2):
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        return normal
    
    triangles = []
    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            # Split quad into two triangles
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # Fan triangulation
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i+1]])
    
    with open(filename, 'wb') as f:
        # Header (80 bytes)
        header = b'Binary STL - Rocket Nozzle' + b'\0' * (80 - 26)
        f.write(header)
        
        # Number of triangles
        f.write(struct.pack('<I', len(triangles)))
        
        # Write triangles
        for tri in triangles:
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            normal = compute_normal(v0, v1, v2)
            
            # Normal vector
            f.write(struct.pack('<fff', *normal))
            # Vertices
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            # Attribute byte count
            f.write(struct.pack('<H', 0))
    
    print(f"Exported {len(triangles)} triangles to {filename}")


def generate_nozzle_mesh(params: NozzleParameters, 
                          n_axial: int = 100, 
                          n_theta: int = 64) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate nozzle inner surface mesh (vertices and faces).
    """
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    
    vertices = []
    faces = []
    
    theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
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


def generate_outer_wall_mesh(params: NozzleParameters,
                              cooling_params: CoolingChannelParameters,
                              n_axial: int = 100,
                              n_theta: int = 64) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate outer wall surface mesh.
    """
    geom = compute_nozzle_geometry(params)
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    
    r_outer = compute_outer_wall_radius(x_profile, r_profile, params, cooling_params, geom)
    
    vertices = []
    faces = []
    
    theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
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
                
                # Reverse winding for outer surface
                faces.append([
                    prev + j,
                    prev + j_next,
                    curr + j_next,
                    curr + j
                ])
    
    return np.array(vertices), faces


def export_geometry_for_viewer(params: NozzleParameters,
                                cooling_params: CoolingChannelParameters,
                                n_axial: int = 80,
                                n_theta: int = 48) -> dict:
    """
    Export all geometry data in JSON format for 3D viewer.
    Includes inner wall, outer wall, and cooling channels (straight or helical).
    """
    geom = compute_nozzle_geometry(params)
    x_profile, r_profile = generate_2d_contour(params, n_axial)
    r_outer = compute_outer_wall_radius(x_profile, r_profile, params, cooling_params, geom)
    
    # Get channel data
    channel_profiles = generate_cooling_channel_profiles(params, cooling_params, n_axial)
    
    # Generate surface data for visualization
    theta_vals = np.linspace(0, 2*np.pi, n_theta).tolist()
    
    # Inner surface
    inner_surface = {
        'x': x_profile.tolist(),
        'r': r_profile.tolist(),
        'theta': theta_vals
    }
    
    # Outer surface
    outer_surface = {
        'x': x_profile.tolist(),
        'r': r_outer.tolist(),
        'theta': theta_vals
    }
    
    # Channel bottom surface (for visualization)
    r_channel_bottom = r_profile + cooling_params.inner_wall_thickness
    channel_bottom_surface = {
        'x': x_profile.tolist(),
        'r': r_channel_bottom.tolist(),
        'theta': theta_vals
    }
    
    # Channel top surface
    channel_top_r = []
    for x, r in zip(x_profile, r_profile):
        ch_geom = compute_channel_geometry_at_x(x, r, params, cooling_params, geom)
        r_top = r + cooling_params.inner_wall_thickness + ch_geom['channel_height']
        channel_top_r.append(r_top)
    
    channel_top_surface = {
        'x': x_profile.tolist(),
        'r': channel_top_r,
        'theta': theta_vals
    }
    
    # Generate helical channel paths if enabled
    helical_data = None
    if cooling_params.channel_type == 'helical':
        x_throat = geom.L_c + geom.L_conv
        throat_idx = np.argmin(np.abs(x_profile - x_throat))
        ch_geom_throat = compute_channel_geometry_at_x(x_throat, r_profile[throat_idx],
                                                        params, cooling_params, geom)
        n_channels = ch_geom_throat['n_channels']
        
        # Generate centerlines for visualization (sample of channels)
        n_sample = min(n_channels, 12)  # Show up to 12 channel paths
        sample_indices = np.linspace(0, n_channels - 1, n_sample, dtype=int)
        
        channel_paths = []
        for ch_idx in sample_indices:
            x_path, y_path, z_path, theta_path = compute_helix_path(
                x_profile, r_profile, ch_idx, n_channels,
                params, cooling_params, geom
            )
            channel_paths.append({
                'channel_index': int(ch_idx),
                'x': x_path.tolist(),
                'y': y_path.tolist(),
                'z': z_path.tolist(),
                'theta': theta_path.tolist()
            })
        
        # Compute helix angle profile
        helix_angles = [compute_helix_angle_at_x(x, params, cooling_params, geom) 
                        for x in x_profile]
        
        helical_data = {
            'enabled': True,
            'n_channels': n_channels,
            'helix_angle_profile': {
                'x': x_profile.tolist(),
                'angle_rad': helix_angles,
                'angle_deg': [np.degrees(a) for a in helix_angles]
            },
            'channel_paths': channel_paths,
            'params': {
                'angle_chamber': cooling_params.helix_angle_chamber,
                'angle_throat': cooling_params.helix_angle_throat,
                'angle_exit': cooling_params.helix_angle_exit,
                'direction': cooling_params.helix_direction
            }
        }
    
    return {
        'params': {
            'R_t': params.R_t,
            'CR': params.CR,
            'L_star': params.L_star,
            'ER': params.ER,
            'theta_n': params.theta_n,
            'theta_e': params.theta_e,
            'L_percent': params.L_percent
        },
        'cooling_params': {
            'inner_wall_thickness': cooling_params.inner_wall_thickness,
            'outer_wall_thickness': cooling_params.outer_wall_thickness,
            'channel_width_throat': cooling_params.channel_width_throat,
            'channel_height_throat': cooling_params.channel_height_throat,
            'rib_width_throat': cooling_params.rib_width_throat,
            'channel_type': cooling_params.channel_type,
            'helix_angle_chamber': cooling_params.helix_angle_chamber,
            'helix_angle_throat': cooling_params.helix_angle_throat,
            'helix_angle_exit': cooling_params.helix_angle_exit,
            'helix_direction': cooling_params.helix_direction
        },
        'geometry': {
            'R_c': geom.R_c,
            'R_e': geom.R_e,
            'L_c': geom.L_c,
            'L_conv': geom.L_conv,
            'L_div': geom.L_div,
            'L_total': geom.L_total,
            'throat_x': geom.L_c + geom.L_conv
        },
        'surfaces': {
            'inner': inner_surface,
            'outer': outer_surface,
            'channel_bottom': channel_bottom_surface,
            'channel_top': channel_top_surface
        },
        'channel_profiles': channel_profiles['channel_profiles'],
        'helical_channels': helical_data
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example parameters
    nozzle_params = NozzleParameters(
        R_t=0.015,       # 15 mm throat radius
        CR=5.0,
        L_star=1.0,
        ER=10.0,
        theta_n=32,
        theta_e=8,
        L_percent=80
    )
    
    # Straight cooling channels
    cooling_params_straight = CoolingChannelParameters(
        inner_wall_thickness=0.001,    # 1 mm
        outer_wall_thickness=0.002,    # 2 mm
        channel_width_throat=0.003,    # 3 mm
        channel_height_throat=0.004,   # 4 mm
        rib_width_throat=0.002,        # 2 mm
        channel_type='straight'
    )
    
    # Helical cooling channels
    cooling_params_helical = CoolingChannelParameters(
        inner_wall_thickness=0.001,    # 1 mm
        outer_wall_thickness=0.002,    # 2 mm
        channel_width_throat=0.003,    # 3 mm
        channel_height_throat=0.004,   # 4 mm
        rib_width_throat=0.002,        # 2 mm
        channel_type='helical',
        helix_angle_chamber=10.0,      # 10° at chamber
        helix_angle_throat=20.0,       # 20° at throat (higher for better heat transfer)
        helix_angle_exit=15.0,         # 15° at exit
        helix_direction=1              # Right-hand helix
    )
    
    print("Generating 3D nozzle geometry...")
    
    # Export for viewer - straight channels
    print("\n--- STRAIGHT CHANNELS ---")
    viewer_data_straight = export_geometry_for_viewer(nozzle_params, cooling_params_straight)
    with open('nozzle_3d_straight.json', 'w') as f:
        json.dump(viewer_data_straight, f, indent=2)
    print("Exported straight channel data to nozzle_3d_straight.json")
    
    # Export for viewer - helical channels
    print("\n--- HELICAL CHANNELS ---")
    viewer_data_helical = export_geometry_for_viewer(nozzle_params, cooling_params_helical)
    with open('nozzle_3d_helical.json', 'w') as f:
        json.dump(viewer_data_helical, f, indent=2)
    print("Exported helical channel data to nozzle_3d_helical.json")
    
    # Generate helical channel mesh
    print("\nGenerating helical channel mesh...")
    helical_mesh = generate_helical_channel_mesh(nozzle_params, cooling_params_helical, n_axial=60)
    print(f"Generated {helical_mesh['n_channels']} helical channels")
    
    # Generate and export STL files
    print("\nGenerating STL meshes...")
    
    # Inner surface
    inner_verts, inner_faces = generate_nozzle_mesh(nozzle_params)
    export_to_stl_binary(inner_verts, inner_faces, 'nozzle_inner.stl')
    
    # Outer surface  
    outer_verts, outer_faces = generate_outer_wall_mesh(nozzle_params, cooling_params_helical)
    export_to_stl_binary(outer_verts, outer_faces, 'nozzle_outer.stl')
    
    # Export a single helical channel as STL for visualization
    if helical_mesh['channels']:
        ch0 = helical_mesh['channels'][0]
        export_to_stl_binary(np.array(ch0['vertices']), ch0['faces'], 
                            'channel_helical_0.stl')
        print("Exported single helical channel to channel_helical_0.stl")
    
    # Print summary
    geom = compute_nozzle_geometry(nozzle_params)
    
    print("\n" + "="*60)
    print("3D NOZZLE GEOMETRY SUMMARY")
    print("="*60)
    print(f"\nNozzle Dimensions:")
    print(f"  Total Length:        {geom.L_total*1000:.1f} mm")
    print(f"  Chamber Radius:      {geom.R_c*1000:.1f} mm")
    print(f"  Throat Radius:       {nozzle_params.R_t*1000:.1f} mm")
    print(f"  Exit Radius:         {geom.R_e*1000:.1f} mm")
    
    print(f"\nHelical Cooling Channel Configuration:")
    print(f"  Number of Channels:  {helical_mesh['n_channels']}")
    print(f"  Inner Wall:          {cooling_params_helical.inner_wall_thickness*1000:.1f} mm")
    print(f"  Channel Width @Throat: {cooling_params_helical.channel_width_throat*1000:.1f} mm")
    print(f"  Channel Height @Throat: {cooling_params_helical.channel_height_throat*1000:.1f} mm")
    print(f"  Rib Width @Throat:   {cooling_params_helical.rib_width_throat*1000:.1f} mm")
    print(f"  Outer Wall:          {cooling_params_helical.outer_wall_thickness*1000:.1f} mm")
    print(f"\n  Helix Angles:")
    print(f"    Chamber:           {cooling_params_helical.helix_angle_chamber:.1f}°")
    print(f"    Throat:            {cooling_params_helical.helix_angle_throat:.1f}°")
    print(f"    Exit:              {cooling_params_helical.helix_angle_exit:.1f}°")
    print(f"  Helix Direction:     {'Right-hand' if cooling_params_helical.helix_direction == 1 else 'Left-hand'}")
    print("="*60)
