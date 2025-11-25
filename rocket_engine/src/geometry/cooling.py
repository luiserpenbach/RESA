import numpy as np
from dataclasses import dataclass
from typing import Literal

from fluids.friction import roughness_Farshad


@dataclass
class CoolingChannelGeometry:
    """Struct to hold resolved geometry arrays (length = N stations)."""
    x_contour: np.ndarray  # Axial position [m]
    radius_contour: np.ndarray  # Inner Chamber radius [m]

    # These are arrays matching the length of x_contour
    channel_width: np.ndarray  # [m]
    channel_height: np.ndarray  # [m]
    rib_width: np.ndarray  # [m]
    wall_thickness: np.ndarray  # [m] (Can vary, usually constant)
    roughness: float
    number_of_channels: int

    @property
    def hydraulic_diameter(self) -> np.ndarray:
        # Dh = 2 * w * h / (w + h) for rectangles
        return 2.0 * self.channel_width * self.channel_height / (
                self.channel_width + self.channel_height
        )

    @property
    def flow_area_per_channel(self) -> np.ndarray:
        return self.channel_width * self.channel_height

    @property
    def coolant_velocity_factor(self) -> np.ndarray:
        """Helper: 1 / (rho * A) factor part of velocity"""
        return 1.0 / self.flow_area_per_channel


class ChannelGeometryGenerator:
    """
    Calculates local channel dimensions along a nozzle contour.
    Supports strategies like 'Constant Rib Width' (Variable Channel) or 'Constant Channel' (Variable Rib).
    """

    def __init__(self,
                 x_contour: np.ndarray,
                 y_contour: np.ndarray,
                 wall_thickness: float = 0.001):
        """
        Args:
            x_contour: Axial positions [m]
            y_contour: Inner radius profile [m]
            wall_thickness: Hot gas wall thickness [m]
        """
        self.xs = x_contour
        self.ys = y_contour
        self.t_wall = wall_thickness

        # Radius of the channel bottom (interface between hot wall and coolant)
        # Note: Usually channels are milled *into* the liner from the outside,
        # or printed.
        # If milled from outside: Channel Bottom Radius = Inner Radius + Wall Thickness
        self.r_channel_base = self.ys + self.t_wall

    def generate_constant_rib(self,
                              num_channels: int,
                              rib_width: float,
                              channel_height: float) -> CoolingChannelGeometry:
        """
        Generates geometry where Rib Width is constant. Channel width varies with radius.

        Args:
            num_channels: Total number of channels
            rib_width: Width of the land between channels [m]
            channel_height: Height of the channel [m] (Constant)

        Returns:
            CoolingChannelGeometry object
        """
        N = num_channels
        w_rib = rib_width

        # Circumference at the channel base = 2 * pi * r
        circumference = 2 * np.pi * self.r_channel_base

        # Total available width for channels = Circumference - (N * RibWidth)
        total_channel_width_avail = circumference - (N * w_rib)

        # Local Channel Width
        w_channels = total_channel_width_avail / N

        # Validation: Check for negative widths (choking)
        min_width = np.min(w_channels)
        if min_width < 0:
            # Find where it fails
            idx_fail = np.argmin(w_channels)
            r_fail = self.r_channel_base[idx_fail]
            raise ValueError(f"Geometry Error: Channels overlap at R={r_fail * 1000:.1f}mm. "
                             f"Too many channels ({N}) or ribs too wide ({w_rib * 1000}mm).")

        return CoolingChannelGeometry(
            x_contour=self.xs,
            radius_contour=self.ys,
            channel_width=w_channels,
            channel_height=np.full_like(self.xs, channel_height),
            rib_width=np.full_like(self.xs, w_rib),
            wall_thickness=np.full_like(self.xs, self.t_wall),
            roughness=10,
            number_of_channels=N
        )

    def calculate_max_channels(self,
                               min_channel_width: float,
                               rib_width: float,
                               at_throat: bool = True) -> int:
        """
        Helper to find how many channels fit at the tightest point (Throat).
        """
        if at_throat:
            r_min = np.min(self.r_channel_base)
        else:
            r_min = self.r_channel_base[0]  # Inlet

        circ = 2 * np.pi * r_min
        # Circ = N * (w_ch + w_rib)
        # N = Circ / (w_ch_min + w_rib)

        return int(np.floor(circ / (min_channel_width + rib_width)))

    def define_by_throat_dimensions(self,
                                    width_at_throat: float,
                                    rib_at_throat: float,
                                    height: float) -> CoolingChannelGeometry:
        """
        Automatically calculates N based on desired dimensions at the throat,
        then generates the full contour maintaining constant rib width.
        """
        # Find Throat Radius (channel base)
        r_throat_base = np.min(self.r_channel_base)

        # Calculate optimal N
        # 2*pi*R = N * (w + rib)
        # N = 2*pi*R / (w + rib)
        circ_throat = 2 * np.pi * r_throat_base
        pitch = width_at_throat + rib_at_throat

        num_channels = int(np.round(circ_throat / pitch))

        print(f"Auto-calculated Channels: {num_channels} (based on Throat R={r_throat_base * 1000:.1f}mm)")

        # Now generate using that N
        return self.generate_constant_rib(num_channels, rib_at_throat, height)

    def define_variable_height(self,
                               base_geometry: CoolingChannelGeometry,
                               height_function: callable) -> CoolingChannelGeometry:
        """
        Advanced: Modifies an existing geometry to have variable channel height.
        Useful for high velocity cooling at throat.

        Args:
            base_geometry: Generated geometry (width profile)
            height_function: Function f(x_pos) -> height [m]
        """
        new_heights = np.array([height_function(x) for x in self.xs])

        base_geometry.channel_height = new_heights
        return base_geometry