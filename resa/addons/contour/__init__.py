"""
RESA Contour Addon
==================
3D geometry generation for rocket engine nozzles and cooling channels.

This module provides tools for:
- Generating 3D nozzle contours (bell nozzle design)
- Creating regenerative cooling channel geometries
- Exporting to STL format for CAD/3D printing
- JSON serialization for data exchange

Example Usage
-------------
>>> from resa.addons.contour import (
...     NozzleParameters,
...     CoolingChannelParameters,
...     Nozzle3DGenerator,
...     CoolingChannel3DGenerator
... )
>>>
>>> # Define nozzle parameters
>>> nozzle_params = NozzleParameters(
...     R_t=0.015,      # 15mm throat radius
...     CR=5.0,         # Contraction ratio
...     L_star=1.0,     # Characteristic length
...     ER=10.0,        # Expansion ratio
...     theta_n=32,     # Initial expansion angle
...     theta_e=8,      # Exit angle
...     L_percent=80    # 80% bell length
... )
>>>
>>> # Generate nozzle
>>> nozzle = Nozzle3DGenerator(nozzle_params)
>>> nozzle.export_inner_stl("nozzle_inner.stl")
>>>
>>> # Define cooling channel parameters
>>> channel_params = CoolingChannelParameters(
...     inner_wall_thickness=0.001,  # 1mm
...     channel_width_throat=0.003,  # 3mm
...     channel_height_throat=0.004, # 4mm
...     channel_type='helical',
...     helix_angle_throat=20.0
... )
>>>
>>> # Generate channels
>>> channels = CoolingChannel3DGenerator(nozzle_params, channel_params)
>>> channels.export_channel_stl("channel_0.stl", channel_index=0)
"""

# Nozzle geometry
from .nozzle_3d import (
    NozzleParameters,
    NozzleGeometry,
    Nozzle3DGenerator,
    compute_nozzle_geometry,
    generate_2d_contour,
    generate_3d_surface_revolution,
    generate_nozzle_mesh,
    generate_outer_wall_mesh,
)

# Cooling channels
from .channels_3d import (
    CoolingChannelParameters,
    CoolingChannel3DGenerator,
    compute_channel_geometry_at_x,
    compute_helix_angle_at_x,
    compute_helix_path,
    generate_straight_channel_mesh,
    generate_helical_channel_mesh,
)

# Export utilities
from .export import (
    export_stl_binary,
    export_stl_ascii,
    export_geometry_json,
    export_mesh_obj,
    load_geometry_json,
)

__all__ = [
    # Nozzle
    "NozzleParameters",
    "NozzleGeometry",
    "Nozzle3DGenerator",
    "compute_nozzle_geometry",
    "generate_2d_contour",
    "generate_3d_surface_revolution",
    "generate_nozzle_mesh",
    "generate_outer_wall_mesh",
    # Channels
    "CoolingChannelParameters",
    "CoolingChannel3DGenerator",
    "compute_channel_geometry_at_x",
    "compute_helix_angle_at_x",
    "compute_helix_path",
    "generate_straight_channel_mesh",
    "generate_helical_channel_mesh",
    # Export
    "export_stl_binary",
    "export_stl_ascii",
    "export_geometry_json",
    "export_mesh_obj",
    "load_geometry_json",
]

__version__ = "1.0.0"
