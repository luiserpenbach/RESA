import numpy as np
from dataclasses import dataclass
from typing import Literal


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

    helix_angle: np.ndarray = None  # [rad] Angle vs Axial direction (0 = Axial, 90 = Hoop)

    def __post_init__(self):
        # Default helix angle to 0 if not provided
        if self.helix_angle is None:
            self.helix_angle = np.zeros_like(self.x_contour)

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
                 wall_thickness: float = 0.001,
                 roughness: float = 10e-6):
        """
        Args:
            x_contour: Axial positions [m]
            y_contour: Inner radius profile [m]
            wall_thickness: Hot gas wall thickness [m]
        """
        self.xs = x_contour
        self.ys = y_contour
        self.t_wall = wall_thickness
        self.roughness = roughness

        # Radius of the channel bottom (interface between hot wall and coolant)
        # If milled from outside: Channel Bottom Radius = Inner Radius + Wall Thickness
        self.r_channel_base = self.ys + self.t_wall

    def generate_constant_rib(self,
                              num_channels: int,
                              rib_width: float,
                              channel_height: float,
                              helix_angle_deg: float = 0.0) -> CoolingChannelGeometry:
        """
        Generates geometry with constant rib width. Supports spiral channels.

        Args:
            num_channels: Total number of channels
            rib_width: Width of the wall between channels [m]
            channel_height: Radial height of the channel [m] (Constant)
            helix_angle_deg: Constant spiral angle in degrees (0 = axial flow)
        Returns:
            CoolingChannelGeometry object
        """
        N = num_channels
        w_rib = rib_width

        # Convert angle to radians
        alpha = np.radians(helix_angle_deg)

        # Effective circumference available for channels cuts perpendicular to flow
        # Circ_normal = 2 * pi * r * cos(alpha)
        # However, usually we define width/rib in the Normal plane (milled cross section).
        # Total Circumference = N * (w_ch / cos(alpha) + w_rib / cos(alpha)) ?
        # Standard approach: The pitch P = 2*pi*r / N is the Hoop Pitch.
        # Normal Pitch P_n = P * cos(alpha)
        # P_n = w_ch_normal + w_rib_normal
        # w_ch_normal = (2*pi*r / N) * cos(alpha) - w_rib_normal

        circumference_hoop = 2 * np.pi * self.r_channel_base
        pitch_hoop = circumference_hoop / N
        pitch_normal = pitch_hoop * np.cos(alpha)
        w_channels = pitch_normal - w_rib

        # Validation
        if np.min(w_channels) < 0:
            idx = np.argmin(w_channels)
            raise ValueError(f"Geometry Error: Channels overlap at R={self.r_channel_base[idx] * 1000:.1f}mm.")

        return CoolingChannelGeometry(
            x_contour=self.xs,
            radius_contour=self.ys,
            channel_width=w_channels,
            channel_height=np.full_like(self.xs, channel_height),
            rib_width=np.full_like(self.xs, w_rib),
            wall_thickness=np.full_like(self.xs, self.t_wall),
            roughness=self.roughness,
            number_of_channels=N,
            helix_angle = np.full_like(self.xs, alpha)
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
                                    height: float,
                                    helix_angle_deg: float = 0.0) -> CoolingChannelGeometry:
        """
        Auto-calculates N based on throat dimensions, including helix angle effect.
        """
        r_throat = np.min(self.r_channel_base)

        # Pitch Normal = Width + Rib
        pitch_normal = width_at_throat + rib_at_throat

        # Pitch Hoop = Pitch Normal / cos(alpha)
        alpha = np.radians(helix_angle_deg)
        pitch_hoop = pitch_normal / np.cos(alpha)

        circ_throat = 2 * np.pi * r_throat
        num_channels = int(np.round(circ_throat / pitch_hoop))

        print(f"Auto-calculated Channels: {num_channels} (Throat R={r_throat * 1000:.1f}mm, Angle={helix_angle_deg}°)")

        return self.generate_constant_rib(num_channels, rib_at_throat, height, helix_angle_deg)

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