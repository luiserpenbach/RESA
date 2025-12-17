import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Wedge
from src.geometry.cooling import CoolingChannelGeometry


def plot_channel_cross_section(geo: CoolingChannelGeometry,
                               station_idx: int,
                               num_channels_show: int = 3,
                               closeout_thickness: float = 0.001):
    """
    Visualizes the 2D cross-section of the cooling channels at a specific axial station.

    Args:
        geo: CoolingChannelGeometry object
        station_idx: Index of the station to plot
        num_channels_show: Number of channels to display in the repeated pattern
        closeout_thickness: Thickness of the outer jacket for visualization [m]
    """

    # 1. Extract Dimensions for this Station
    # All dimensions in mm for plotting
    x_pos = geo.x_contour[station_idx] * 1000
    w_ch = geo.channel_width[station_idx] * 1000
    h_ch = geo.channel_height[station_idx] * 1000
    w_rib = geo.rib_width[station_idx] * 1000
    t_wall = geo.wall_thickness[station_idx] * 1000
    t_closeout = closeout_thickness * 1000
    radius = geo.radius_contour[station_idx] * 1000

    pitch = w_ch + w_rib
    total_width = num_channels_show * pitch + w_rib  # End with a rib
    total_height = t_wall + h_ch + t_closeout

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Draw Solid Material (The "Liner")
    # We draw the main block first, then "cut out" the channels?
    # Or draw parts. Let's draw parts.

    # Color Scheme
    color_metal = '#B87333'  # Copper-ish
    color_coolant = '#4db6ac'  # Teal-ish
    color_gas = '#ffccbc'  # Light Orange

    # A. Draw Hot Wall (Bottom Layer)
    # Rectangle from (0, 0) to (TotalWidth, t_wall)
    hot_wall = Rectangle((0, 0), total_width, t_wall,
                         facecolor=color_metal, edgecolor='k', label='Liner Material')
    ax.add_patch(hot_wall)

    # B. Draw Ribs
    # Ribs start at y = t_wall, height = h_ch
    for i in range(num_channels_show + 1):
        x_rib = i * pitch
        rib = Rectangle((x_rib, t_wall), w_rib, h_ch,
                        facecolor=color_metal, edgecolor='k')
        ax.add_patch(rib)

    # C. Draw Closeout (Top Layer)
    # Starts at y = t_wall + h_ch
    closeout = Rectangle((0, t_wall + h_ch), total_width, t_closeout,
                         facecolor='#A9A9A9', edgecolor='k', hatch='///', label='Closeout/Jacket')
    ax.add_patch(closeout)

    # D. Draw Coolant (The Voids)
    for i in range(num_channels_show):
        x_ch = i * pitch + w_rib
        channel = Rectangle((x_ch, t_wall), w_ch, h_ch,
                            facecolor=color_coolant, alpha=0.6, edgecolor='k', label='Coolant' if i == 0 else None)
        ax.add_patch(channel)

        # Add dimensions text
        if i == 1:  # Center channel
            # Width text
            ax.annotate(f"{w_ch:.2f} mm", (x_ch + w_ch / 2, t_wall + h_ch / 2),
                        ha='center', va='center', fontsize=9, color='white', weight='bold')
            # Height text
            ax.annotate(f"{h_ch:.2f}", (x_ch - 0.2, t_wall + h_ch / 2),
                        ha='right', va='center', fontsize=8, rotation=90)

    # E. Annotate Thicknesses
    # Wall Thickness arrow
    ax.annotate("", xy=(-0.5, 0), xytext=(-0.5, t_wall), arrowprops=dict(arrowstyle='<->'))
    ax.text(-0.6, t_wall / 2, f"Wall\n{t_wall:.2f}mm", ha='right', va='center', fontsize=9)

    # Rib Width
    center_rib_x = 1 * pitch + w_rib / 2
    ax.annotate(f"Rib\n{w_rib:.2f}mm", (center_rib_x, t_wall + h_ch + t_closeout + 0.5),
                ha='center', va='bottom', fontsize=9, arrowprops=dict(arrowstyle='->', relpos=(0.5, 0)))

    # 4. Styling
    ax.set_xlim(-2, total_width + 1)
    ax.set_ylim(-1, total_height + 2)
    ax.set_aspect('equal')
    ax.set_xlabel("Tangential Width [mm] (Flattened)")
    ax.set_ylabel("Radial Thickness [mm]")
    ax.set_title(f"Channel Cross-Section at X = {x_pos:.1f} mm\n(R = {radius:.1f} mm)")

    # Custom Legend
    handles = [
        Patch(facecolor=color_metal, edgecolor='k', label='Chamber Liner'),
        Patch(facecolor=color_coolant, alpha=0.6, edgecolor='k', label='Coolant (N2O)'),
        Patch(facecolor='#A9A9A9', hatch='///', edgecolor='k', label='Electroplated Jacket')
    ]
    ax.legend(handles=handles, loc='upper right')

    # Gas Flow indication
    ax.text(total_width / 2, -0.8, "HOT GAS FLOW IS HERE", ha='center', va='center',
            color='red', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.show()



def plot_channel_cross_section_radial(geo: CoolingChannelGeometry,
                                      station_idx: int,
                                      closeout_thickness: float = 0.001,
                                      sector_angle: float = 360.0,
                                      show: bool = True):
    """
    Visualizes the channel cross-section in true Radial (Polar) coordinates.

    Args:
        geo: CoolingChannelGeometry object
        station_idx: Index of the station to plot
        closeout_thickness: Thickness of the outer jacket [m]
        sector_angle: Angle of the sector to plot [degrees].
                      360 for full ring, 90 for quarter, etc.
    """

    # 1. Extract Dimensions (in mm)
    R_inner = geo.radius_contour[station_idx] * 1000
    t_wall = geo.wall_thickness[station_idx] * 1000
    w_ch = geo.channel_width[station_idx] * 1000
    h_ch = geo.channel_height[station_idx] * 1000
    w_rib = geo.rib_width[station_idx] * 1000
    t_closeout = closeout_thickness * 1000

    num_channels = geo.number_of_channels

    # Calculate Radii
    R_channel_base = R_inner + t_wall
    R_channel_top = R_channel_base + h_ch
    R_outer = R_channel_top + t_closeout

    # Calculate Angles (in degrees)
    # Note: We assume w_ch and w_rib are arc lengths at R_channel_base
    pitch_arc = w_ch + w_rib

    # Angle subtended by one channel pitch
    theta_pitch = 360.0 / num_channels

    # Angle subtended by the channel open width
    # fraction = w_ch / (w_ch + w_rib)
    theta_ch = theta_pitch * (w_ch / pitch_arc)
    theta_rib = theta_pitch - theta_ch

    # Determine how many channels to draw based on sector_angle
    if sector_angle >= 360:
        channels_to_draw = num_channels
        sector_angle = 360.0
    else:
        channels_to_draw = int(np.ceil(sector_angle / theta_pitch))

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color Scheme
    color_metal = '#B87333'  # Copper
    color_coolant = '#4db6ac'  # Teal
    color_closeout = '#A9A9A9'  # Grey

    # A. Draw the Full Annulus Background (The Liner Material + Closeout)
    # We draw from R_inner to R_outer, then we will "paint" the channels on top
    # or draw layer by layer.

    # Layer 1: Inner Liner (Solid Metal)
    # Wedge from R_inner to R_channel_base
    liner_base = Wedge((0, 0), R_channel_base, 0, sector_angle, width=t_wall,
                       facecolor=color_metal, edgecolor='k')
    ax.add_patch(liner_base)

    # Layer 2: The Channel/Rib Layer
    # We draw the "Ring" of Rib material first, then overlay channels?
    # Or draw Rib Wedges. Drawing Rib Wedges is cleaner.

    # Draw a "Background" ring for the coolant layer (representing ribs everywhere)
    # then we can't easily cut holes.
    # Better: Draw discrete Rib Wedges.

    start_angle = 0.0
    for i in range(channels_to_draw):
        # 1. Channel Wedge (Coolant)
        # Starts at start_angle
        wedge_ch = Wedge((0, 0), R_channel_top, start_angle, start_angle + theta_ch,
                         width=h_ch, facecolor=color_coolant, alpha=0.7, edgecolor=None)
        ax.add_patch(wedge_ch)

        # 2. Rib Wedge (Metal)
        # Starts at start_angle + theta_ch
        wedge_rib = Wedge((0, 0), R_channel_top, start_angle + theta_ch, start_angle + theta_pitch,
                          width=h_ch, facecolor=color_metal, edgecolor='k', linewidth=0.5)
        ax.add_patch(wedge_rib)

        start_angle += theta_pitch

    # Layer 3: Closeout / Jacket
    # Wedge from R_channel_top to R_outer
    jacket = Wedge((0, 0), R_outer, 0, sector_angle, width=t_closeout,
                   facecolor=color_closeout, hatch='///', edgecolor='k')
    ax.add_patch(jacket)

    # B. Annotations
    # Draw Inner Radius Circle (Ghost)
    if sector_angle < 360:
        # Draw boundary lines for the sector
        pass

        # Annotate one channel if zoomed in, or just general info
    # Center text
    ax.text(0, 0, f"Station X={geo.x_contour[station_idx] * 1000:.1f}mm\nN={num_channels}",
            ha='center', va='center', fontweight='bold')

    # Dimensions on the side
    ax.text(R_outer * 1.1, 0,
            f"ID: {2 * R_inner:.1f} mm\nOD: {2 * R_outer:.1f} mm",
            va='center')

    # 3. Styling
    limit = R_outer * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes for polar view

    # Legend
    handles = [
        Patch(facecolor=color_metal, edgecolor='k', label='Liner (Cu)'),
        Patch(facecolor=color_coolant, alpha=0.7, label='Coolant'),
        Patch(facecolor=color_closeout, hatch='///', edgecolor='k', label='Closeout')
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.title(f"Radial Cross-Section\nStation {station_idx}", y=1.02)
    plt.tight_layout()
    if show:
        plt.show()

    return fig  # Return the figure object!