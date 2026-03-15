"""Cooling Channel Geometry Generator for RESA."""
from typing import Callable, Optional

import numpy as np

from resa.core.results import CoolingChannelGeometry, NozzleGeometry


class ChannelGeometryGenerator:
    """
    Generates cooling channel geometry along nozzle contour.

    Supports:
    - Constant rib width (variable channel width)
    - Spiral/helical channels
    - Variable height channels
    """

    def generate(
        self,
        nozzle_geometry: NozzleGeometry,
        channel_width_throat: float,
        channel_height: float,
        rib_width_throat: float,
        wall_thickness: float = 0.001,
        roughness: float = 10e-6,
        helix_angle_deg: float = 0.0,
        channel_type: str = "rectangular",
        taper_angle_deg: float = 10.0,
        num_channels_override: Optional[int] = None,
    ) -> CoolingChannelGeometry:
        """
        Generate cooling channel geometry.

        Args:
            nozzle_geometry: NozzleGeometry from nozzle generator
            channel_width_throat: Channel width at throat [m]
            channel_height: Channel height (radial depth) [m]
            rib_width_throat: Rib width at throat [m]
            wall_thickness: Hot wall thickness [m]
            roughness: Surface roughness [m]
            helix_angle_deg: Helix angle for spiral channels [degrees]
            channel_type: 'rectangular' or 'trapezoidal'
            taper_angle_deg: Sidewall taper angle for trapezoidal channels [deg]
            num_channels_override: Override automatic channel count

        Returns:
            CoolingChannelGeometry with channel dimensions
        """
        x = nozzle_geometry.x_full
        y = nozzle_geometry.y_full
        Rt = nozzle_geometry.throat_radius

        # Radius at channel base (outside of hot wall)
        r_channel_base = y + wall_thickness

        # Calculate number of channels based on throat geometry
        r_throat = Rt + wall_thickness
        alpha = np.radians(helix_angle_deg)
        pitch_normal = channel_width_throat + rib_width_throat
        pitch_hoop = pitch_normal / np.cos(alpha) if alpha != 0 else pitch_normal
        circ_throat = 2 * np.pi * r_throat

        if num_channels_override is not None and num_channels_override > 0:
            num_channels = num_channels_override
        else:
            num_channels = int(np.round(circ_throat / pitch_hoop))

        # Calculate channel width along contour (constant rib width)
        circumference_hoop = 2 * np.pi * r_channel_base
        pitch_hoop_local = circumference_hoop / num_channels
        pitch_normal_local = pitch_hoop_local * np.cos(alpha)
        channel_width = pitch_normal_local - rib_width_throat

        # For trapezoidal channels, compute effective hydraulic width
        # The bottom width is narrower than top width due to taper
        if channel_type == "trapezoidal" and taper_angle_deg > 0:
            taper_rad = np.radians(taper_angle_deg)
            # Width at bottom of channel (narrower)
            width_reduction = 2.0 * channel_height * np.tan(taper_rad)
            channel_width_bottom = channel_width - width_reduction
            # Use average width for hydraulic calculations
            channel_width = (channel_width + np.maximum(channel_width_bottom, 1e-6)) / 2.0

        # Ensure positive channel width
        channel_width = np.maximum(channel_width, 1e-6)

        return CoolingChannelGeometry(
            x=x,
            y=y,
            channel_width=channel_width,
            channel_height=np.full_like(x, channel_height),
            rib_width=np.full_like(x, rib_width_throat),
            wall_thickness=np.full_like(x, wall_thickness),
            roughness=roughness,
            num_channels=num_channels,
            helix_angle=np.full_like(x, alpha)
        )

    def generate_variable_height(
        self,
        base_geometry: CoolingChannelGeometry,
        height_function: Callable[[float], float]
    ) -> CoolingChannelGeometry:
        """
        Modify geometry to have variable channel height.

        Args:
            base_geometry: Base channel geometry
            height_function: Function f(x_position) -> height [m]

        Returns:
            Modified CoolingChannelGeometry
        """
        new_heights = np.array([height_function(xi) for xi in base_geometry.x])

        return CoolingChannelGeometry(
            x=base_geometry.x,
            y=base_geometry.y,
            channel_width=base_geometry.channel_width,
            channel_height=new_heights,
            rib_width=base_geometry.rib_width,
            wall_thickness=base_geometry.wall_thickness,
            roughness=base_geometry.roughness,
            num_channels=base_geometry.num_channels,
            helix_angle=base_geometry.helix_angle
        )
