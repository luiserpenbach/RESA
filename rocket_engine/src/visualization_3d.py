import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rocket_engine.src.geometry.cooling import CoolingChannelGeometry


def plot_coolant_channels_3d(geo: CoolingChannelGeometry,
                             num_channels_to_show: int = 5,
                             resolution: int = 50):
    """
    Visualizes the COOLANT DOMAIN (the fluid) as solid 3D tubes.
    This helps you see the expanding/contracting channel shapes.

    Args:
        geo: CoolingChannelGeometry object
        num_channels_to_show: How many adjacent channels to render
        resolution: Downsampling factor for axial smoothness
    """

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Downsample Data
    step = max(1, len(geo.x_contour) // resolution)
    xs = geo.x_contour[::step] * 1000  # mm

    # Dimensions at each station
    rs_inner = geo.radius_contour[::step] * 1000
    t_wall = geo.wall_thickness[::step] * 1000
    w_ch = geo.channel_width[::step] * 1000
    h_ch = geo.channel_height[::step] * 1000
    w_rib = geo.rib_width[::step] * 1000

    # Radial bounds of the fluid
    r_bottom = rs_inner + t_wall
    r_top = r_bottom + h_ch

    # 2. Calculate Angular Positions
    # We assume uniform distribution around the clock
    # Pitch (Arc Length) approx = w_ch + w_rib
    # But strictly, Pitch_Angle = 360 / N

    N = geo.number_of_channels
    pitch_angle = 2 * np.pi / N  # Radians

    # We need the angular width of the channel at each station
    # theta_ch = pitch_angle * (w_ch / (w_ch + w_rib))
    # Note: w_ch and w_rib change along X, so theta_ch changes along X!
    # This gives the channels their "twist" or expansion.

    total_arc = w_ch + w_rib
    theta_widths = pitch_angle * (w_ch / total_arc)

    print(f"Rendering {num_channels_to_show} Coolant Channels...")

    # Color Palette for multiple channels
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, num_channels_to_show))

    # 3. Generate Tubes
    for k in range(num_channels_to_show):
        # Center angle for this channel
        theta_center_0 = k * pitch_angle

        # We need to construct the 4 walls of the rectangular tube along X
        # Arrays to hold coordinate grids
        X_grid = np.tile(xs, (2, 1))

        # Calculate angular bounds at every X
        # Center stays at k * pitch (assuming no swirl for now)
        # If you had swirl, theta_center would be f(x)
        theta_center = theta_center_0

        theta_left = theta_center - theta_widths / 2
        theta_right = theta_center + theta_widths / 2

        # --- FACE 1: BOTTOM (Hot Side) ---
        # Fixed R = r_bottom, Theta varies left to right
        Y_bot = np.array([r_bottom * np.cos(theta_left), r_bottom * np.cos(theta_right)])
        Z_bot = np.array([r_bottom * np.sin(theta_left), r_bottom * np.sin(theta_right)])
        ax.plot_surface(X_grid, Y_bot, Z_bot, color=colors[k], alpha=0.8, shade=True)

        # --- FACE 2: TOP (Cold Side) ---
        # Fixed R = r_top
        Y_top = np.array([r_top * np.cos(theta_left), r_top * np.cos(theta_right)])
        Z_top = np.array([r_top * np.sin(theta_left), r_top * np.sin(theta_right)])
        ax.plot_surface(X_grid, Y_top, Z_top, color=colors[k], alpha=0.6, shade=True)

        # --- FACE 3: LEFT WALL (Rib Side 1) ---
        # Theta fixed at theta_left, R varies bottom to top
        Y_left = np.array([r_bottom * np.cos(theta_left), r_top * np.cos(theta_left)])
        Z_left = np.array([r_bottom * np.sin(theta_left), r_top * np.sin(theta_left)])
        ax.plot_surface(X_grid, Y_left, Z_left, color=colors[k], alpha=0.4, shade=True)

        # --- FACE 4: RIGHT WALL (Rib Side 2) ---
        # Theta fixed at theta_right
        Y_right = np.array([r_bottom * np.cos(theta_right), r_top * np.cos(theta_right)])
        Z_right = np.array([r_bottom * np.sin(theta_right), r_top * np.sin(theta_right)])
        ax.plot_surface(X_grid, Y_right, Z_right, color=colors[k], alpha=0.4, shade=True)

    # 4. Context: Draw Inner Liner (Ghost)
    # Just a wireframe or transparent surface to show where the engine is
    theta_liner = np.linspace(-pitch_angle, num_channels_to_show * pitch_angle, 20)
    X_liner, T_liner = np.meshgrid(xs, theta_liner)
    R_liner = np.tile(rs_inner, (len(theta_liner), 1)).T
    # Fix shape mismatch: R_liner needs to match X_liner (len(xs), len(theta))
    # Re-mesh
    X_liner, T_liner = np.meshgrid(xs, theta_liner, indexing='ij')
    R_liner = np.tile(rs_inner[:, np.newaxis], (1, len(theta_liner)))

    Y_liner = R_liner * np.cos(T_liner)
    Z_liner = R_liner * np.sin(T_liner)

    ax.plot_surface(X_liner, Y_liner, Z_liner, color='#B87333', alpha=0.1)

    # 5. Styling
    ax.set_xlabel('Axial X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title(f"3D Coolant Domain ({num_channels_to_show} Channels)")

    # Axis Equal Hack
    max_range = np.array([xs.max() - xs.min(), r_top.max()]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = r_top.max() * 0.5 if num_channels_to_show > N / 2 else 0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.view_init(elev=30, azim=-60)
    plt.tight_layout()
    plt.show()


def plot_engine_3d(geo: CoolingChannelGeometry,
                   closeout_thickness: float = 0.001,
                   sector_angle: float = 90.0,
                   resolution: int = 50):
    """
    Generates a 3D visualization of the engine cooling jacket.

    Args:
        geo: CoolingChannelGeometry object
        closeout_thickness: Thickness of the outer jacket [m]
        sector_angle: Wedge angle to display (e.g., 90 deg cutaway).
        resolution: Number of axial points to skip for faster plotting (downsampling).
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample arrays for performance
    # We take every Nth point to keep the mesh light
    step = max(1, len(geo.x_contour) // resolution)
    xs = geo.x_contour[::step] * 1000  # mm
    rs_inner = geo.radius_contour[::step] * 1000  # mm

    # Calculate Outer Radius (Top of ribs + Closeout)
    # R_outer = R_inner + t_wall + h_ch + t_closeout
    t_wall = geo.wall_thickness[::step] * 1000
    h_ch = geo.channel_height[::step] * 1000
    rs_outer = rs_inner + t_wall + h_ch + (closeout_thickness * 1000)

    # --- Generate Mesh Grid ---
    # Theta grid (0 to sector_angle)
    theta = np.linspace(0, np.radians(sector_angle), 30)

    # Create 2D grids
    Theta, X = np.meshgrid(theta, xs)

    # 1. Inner Surface (Hot Wall)
    # R is constant at a given X, but X varies
    # We need to broadcast R array to the shape of Theta
    # R_grid has shape (len(xs), len(theta))
    R_inner_grid = rs_inner[:, np.newaxis] * np.ones_like(Theta)

    Y_inner = R_inner_grid * np.cos(Theta)
    Z_inner = R_inner_grid * np.sin(Theta)

    # Plot Inner Surface (Copper Color)
    surf_inner = ax.plot_surface(X, Y_inner, Z_inner, color='#B87333', alpha=1.0,
                                 rcount=resolution, ccount=30, shade=True)

    # 2. Outer Surface (Closeout)
    R_outer_grid = rs_outer[:, np.newaxis] * np.ones_like(Theta)
    Y_outer = R_outer_grid * np.cos(Theta)
    Z_outer = R_outer_grid * np.sin(Theta)

    # Plot Outer Surface (Grey, semi-transparent)
    surf_outer = ax.plot_surface(X, Y_outer, Z_outer, color='gray', alpha=0.3,
                                 rcount=resolution, ccount=30, shade=True)

    # 3. Visualizing Channels (Optional - Wireframe)
    # Drawing 3D voids is hard in matplotlib.
    # Instead, we draw lines representing the Ribs on the Inner surface?
    # Or simply let the gap between surfaces represent the channels.

    # Let's verify dimensions visually by capping the ends
    # Draw a line at the inlet and exit face
    for i in [0, -1]:
        x_cap = xs[i]
        r_in = rs_inner[i]
        r_out = rs_outer[i]

        # Draw spokes
        for th in np.linspace(0, np.radians(sector_angle), 5):
            ax.plot([x_cap, x_cap],
                    [r_in * np.cos(th), r_out * np.cos(th)],
                    [r_in * np.sin(th), r_out * np.sin(th)], 'k-', lw=1)

    # 4. Styling
    ax.set_xlabel('Axial Length X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title(f'3D Engine Cutaway ({sector_angle}° Sector)')

    # Equal Aspect Ratio Hack for 3D
    # Matplotlib 3D doesn't support 'equal' natively well.
    # We construct a bounding box.
    max_range = np.array([xs.max() - xs.min(), rs_outer.max(), rs_outer.max()]).max() / 2.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = rs_outer.max() * 0.5
    mid_z = rs_outer.max() * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # View Angle
    ax.view_init(elev=20, azim=-45)

    plt.tight_layout()
    plt.show()


# --- Advanced: Export to STL for CAD ---
def export_to_stl(geo: CoolingChannelGeometry, filename: str = "engine.stl"):
    """
    Exports the Inner Liner geometry to an STL file for CAD import.
    Requires numpy-stl library (pip install numpy-stl).
    """
    try:
        from stl import mesh
    except ImportError:
        print("Error: 'numpy-stl' library not found. Run: pip install numpy-stl")
        return

    # Create a simplified revolution
    # This creates the "Solid of Revolution" of the contour
    # Defining vertices is complex here, but here is a simple implementation
    # that revolves the contour 360 degrees.

    print("STL Export not fully implemented yet (Requires robust meshing logic).")
    print("Recommendation: Use the CSV profile output and 'Revolve' in Fusion360/SolidWorks.")