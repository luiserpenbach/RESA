"""
3D WebGL Engine Visualization using Plotly.

Provides interactive 3D visualizations for:
- Nozzle geometry (surface of revolution)
- Cooling channels (3D tube representation)
- Full engine assembly with cutaway views
- Temperature/heat flux color mapping

All visualizations use Plotly's WebGL-accelerated 3D graphics
for smooth interactivity and easy HTML export.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Tuple, List, Union, TYPE_CHECKING

from resa.visualization.themes import PlotTheme, EngineeringTheme, DarkTheme, DEFAULT_THEME

if TYPE_CHECKING:
    from rocket_engine.src.geometry.nozzle import NozzleGeometryData
    from rocket_engine.src.geometry.cooling import CoolingChannelGeometry


class Engine3DViewer:
    """
    Interactive 3D visualization of rocket engine geometry using Plotly WebGL.

    Supports:
    - Surface of revolution from 2D nozzle contours
    - 3D cooling channel rendering
    - Temperature/heat flux color mapping
    - Light and dark themes
    - HTML export for reports and embedding

    Example:
        viewer = Engine3DViewer()
        fig = viewer.render_nozzle(nozzle_geometry)
        fig.show()  # Interactive 3D display
        html = viewer.to_html(fig)  # Export for embedding
    """

    def __init__(
        self,
        theme: Optional[PlotTheme] = None,
        dark_mode: bool = False
    ):
        """
        Initialize the 3D viewer.

        Args:
            theme: Custom PlotTheme for styling. If None, uses default.
            dark_mode: If True, uses dark theme styling.
        """
        if theme is not None:
            self.theme = theme
        elif dark_mode:
            self.theme = DarkTheme()
        else:
            self.theme = DEFAULT_THEME

        self.dark_mode = dark_mode

        # Default camera settings for engine viewing
        self._default_camera = dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        )

    def _create_surface_of_revolution(
        self,
        x_profile: np.ndarray,
        r_profile: np.ndarray,
        n_theta: int = 60,
        theta_start: float = 0.0,
        theta_end: float = 2 * np.pi
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 3D mesh coordinates by revolving a 2D profile around the X axis.

        Args:
            x_profile: Axial coordinates of the 2D contour
            r_profile: Radial coordinates of the 2D contour
            n_theta: Number of angular divisions for revolution
            theta_start: Starting angle for partial revolution (radians)
            theta_end: Ending angle for partial revolution (radians)

        Returns:
            Tuple of (X, Y, Z) coordinate meshes
        """
        theta = np.linspace(theta_start, theta_end, n_theta)

        # Create meshgrid: X varies along rows, Theta varies along columns
        X_mesh, Theta_mesh = np.meshgrid(x_profile, theta, indexing='ij')
        R_mesh = np.tile(r_profile[:, np.newaxis], (1, n_theta))

        # Convert cylindrical (X, R, Theta) to Cartesian (X, Y, Z)
        Y_mesh = R_mesh * np.cos(Theta_mesh)
        Z_mesh = R_mesh * np.sin(Theta_mesh)

        return X_mesh, Y_mesh, Z_mesh

    def _mesh3d_from_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        color: str = '#B87333',
        opacity: float = 1.0,
        name: str = '',
        intensity: Optional[np.ndarray] = None,
        colorscale: Optional[List] = None,
        showscale: bool = False,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        colorbar_title: str = ''
    ) -> go.Mesh3d:
        """
        Create a Mesh3d trace from a surface grid.

        Converts a parametric surface (X, Y, Z grids) into triangular mesh
        suitable for Plotly's Mesh3d visualization.

        Args:
            X, Y, Z: 2D coordinate arrays defining the surface
            color: Uniform surface color (used if intensity is None)
            opacity: Surface transparency (0.0 to 1.0)
            name: Trace name for legend
            intensity: Optional array for color mapping by value
            colorscale: Plotly colorscale for intensity mapping
            showscale: Whether to show colorbar
            cmin, cmax: Color scale range limits
            colorbar_title: Title for the colorbar

        Returns:
            Plotly Mesh3d trace
        """
        # Flatten coordinates
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()

        ni, nj = X.shape

        # Build triangle indices
        # Each quad cell (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j)
        # is split into 2 triangles
        i_indices = []
        j_indices = []
        k_indices = []

        for i in range(ni - 1):
            for j in range(nj - 1):
                # Vertex indices in flat array
                p00 = i * nj + j
                p01 = i * nj + (j + 1)
                p10 = (i + 1) * nj + j
                p11 = (i + 1) * nj + (j + 1)

                # Triangle 1: p00 - p01 - p11
                i_indices.append(p00)
                j_indices.append(p01)
                k_indices.append(p11)

                # Triangle 2: p00 - p11 - p10
                i_indices.append(p00)
                j_indices.append(p11)
                k_indices.append(p10)

        # Build mesh arguments
        mesh_args = dict(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            i=i_indices,
            j=j_indices,
            k=k_indices,
            opacity=opacity,
            name=name,
            showlegend=bool(name),
            flatshading=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5,
                fresnel=0.2
            ),
            lightposition=dict(
                x=1000,
                y=1000,
                z=1000
            )
        )

        if intensity is not None:
            # Color by intensity values
            intensity_flat = intensity.flatten() if intensity.ndim > 1 else np.tile(intensity, len(x_flat) // len(intensity) + 1)[:len(x_flat)]
            mesh_args['intensity'] = intensity_flat
            mesh_args['colorscale'] = colorscale or self.theme.temperature_colorscale
            mesh_args['showscale'] = showscale
            if cmin is not None:
                mesh_args['cmin'] = cmin
            if cmax is not None:
                mesh_args['cmax'] = cmax
            if showscale:
                mesh_args['colorbar'] = dict(
                    title=dict(
                        text=colorbar_title,
                        side='right'
                    ),
                    thickness=20,
                    len=0.6
                )
        else:
            # Uniform color
            mesh_args['color'] = color

        return go.Mesh3d(**mesh_args)

    def _create_wireframe(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        color: str = 'black',
        width: float = 1.0,
        n_axial: int = 10,
        n_circ: int = 8,
        name: str = ''
    ) -> List[go.Scatter3d]:
        """
        Create wireframe lines from a surface for visual clarity.

        Args:
            X, Y, Z: Surface coordinate meshes
            color: Line color
            width: Line width
            n_axial: Number of axial lines to draw
            n_circ: Number of circumferential lines to draw
            name: Base name for traces

        Returns:
            List of Scatter3d traces for the wireframe
        """
        traces = []
        ni, nj = X.shape

        # Axial lines (along rows, at selected column indices)
        j_indices = np.linspace(0, nj - 1, n_circ, dtype=int)
        for idx, j in enumerate(j_indices):
            traces.append(go.Scatter3d(
                x=X[:, j],
                y=Y[:, j],
                z=Z[:, j],
                mode='lines',
                line=dict(color=color, width=width),
                name=f'{name} axial' if idx == 0 else '',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Circumferential lines (along columns, at selected row indices)
        i_indices = np.linspace(0, ni - 1, n_axial, dtype=int)
        for idx, i in enumerate(i_indices):
            traces.append(go.Scatter3d(
                x=X[i, :],
                y=Y[i, :],
                z=Z[i, :],
                mode='lines',
                line=dict(color=color, width=width),
                name=f'{name} circ' if idx == 0 else '',
                showlegend=False,
                hoverinfo='skip'
            ))

        return traces

    def render_nozzle(
        self,
        nozzle_geometry: 'NozzleGeometryData',
        sector_angle: float = 360.0,
        show_wireframe: bool = True,
        n_theta: int = 60,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Render 3D nozzle geometry as a surface of revolution.

        Args:
            nozzle_geometry: NozzleGeometryData object with x_full, y_full properties
            sector_angle: Angular extent to render (degrees). 360 for full,
                         90-180 for cutaway views.
            show_wireframe: If True, overlay wireframe lines for clarity
            n_theta: Angular resolution (number of segments around circumference)
            title: Optional custom title

        Returns:
            Plotly Figure with 3D nozzle visualization
        """
        # Extract contour data (convert mm to display units)
        x_profile = nozzle_geometry.x_full
        r_profile = nozzle_geometry.y_full

        # Create surface of revolution
        theta_end = np.radians(sector_angle)
        X, Y, Z = self._create_surface_of_revolution(
            x_profile, r_profile,
            n_theta=n_theta,
            theta_end=theta_end
        )

        # Create figure
        fig = go.Figure()

        # Add main surface
        surface = self._mesh3d_from_surface(
            X, Y, Z,
            color=self.theme.copper,
            opacity=1.0,
            name='Nozzle Wall'
        )
        fig.add_trace(surface)

        # Add wireframe
        if show_wireframe:
            wireframe = self._create_wireframe(
                X, Y, Z,
                color='rgba(0,0,0,0.3)' if not self.dark_mode else 'rgba(255,255,255,0.3)',
                width=1.0,
                n_axial=15,
                n_circ=12,
                name='Wireframe'
            )
            for trace in wireframe:
                fig.add_trace(trace)

        # Add contour outline on cut plane (for sectional views)
        if sector_angle < 360:
            # Draw profile line at theta=0
            fig.add_trace(go.Scatter3d(
                x=x_profile,
                y=r_profile,
                z=np.zeros_like(x_profile),
                mode='lines',
                line=dict(color='black', width=3),
                name='Inner Contour',
                hovertemplate='X: %{x:.1f} mm<br>R: %{y:.2f} mm<extra></extra>'
            ))
            # Mirror at theta_end
            fig.add_trace(go.Scatter3d(
                x=x_profile,
                y=r_profile * np.cos(theta_end),
                z=r_profile * np.sin(theta_end),
                mode='lines',
                line=dict(color='black', width=3),
                name='',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Configure layout
        self._apply_3d_layout(
            fig,
            title=title or 'Nozzle Geometry (3D)',
            x_label='Axial Position [mm]',
            y_label='Y [mm]',
            z_label='Z [mm]'
        )

        return fig

    def render_channels(
        self,
        channel_geometry: 'CoolingChannelGeometry',
        n_channels_to_show: int = 4,
        show_liner: bool = True,
        resolution: int = 50,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Render cooling channels as 3D tubes.

        Visualizes the coolant flow domain as rectangular cross-section tubes
        following the nozzle contour.

        Args:
            channel_geometry: CoolingChannelGeometry object
            n_channels_to_show: Number of adjacent channels to render
            show_liner: If True, show semi-transparent inner liner
            resolution: Axial point downsampling for performance
            title: Optional custom title

        Returns:
            Plotly Figure with 3D channel visualization
        """
        geo = channel_geometry

        # Downsample for performance
        step = max(1, len(geo.x_contour) // resolution)
        xs = geo.x_contour[::step] * 1000  # Convert to mm
        rs_inner = geo.radius_contour[::step] * 1000
        t_wall = geo.wall_thickness[::step] * 1000
        w_ch = geo.channel_width[::step] * 1000
        h_ch = geo.channel_height[::step] * 1000
        w_rib = geo.rib_width[::step] * 1000

        # Channel bounds
        r_bottom = rs_inner + t_wall
        r_top = r_bottom + h_ch

        N = geo.number_of_channels
        pitch_angle = 2 * np.pi / N

        # Angular width of channel (varies with position)
        total_arc = w_ch + w_rib
        theta_widths = pitch_angle * (w_ch / total_arc)

        fig = go.Figure()

        # Generate color palette for channels
        colors = self._get_channel_colors(n_channels_to_show)

        # Render each channel
        for k in range(n_channels_to_show):
            theta_center = k * pitch_angle
            theta_left = theta_center - theta_widths / 2
            theta_right = theta_center + theta_widths / 2

            channel_traces = self._create_channel_tube(
                xs, r_bottom, r_top, theta_left, theta_right,
                color=colors[k],
                name=f'Channel {k+1}' if k < 3 else '',
                show_legend=(k == 0)
            )
            for trace in channel_traces:
                fig.add_trace(trace)

        # Add semi-transparent liner
        if show_liner:
            theta_extent = (n_channels_to_show + 0.5) * pitch_angle
            X_liner, Y_liner, Z_liner = self._create_surface_of_revolution(
                xs, rs_inner,
                n_theta=40,
                theta_start=-pitch_angle * 0.5,
                theta_end=theta_extent
            )

            liner = self._mesh3d_from_surface(
                X_liner, Y_liner, Z_liner,
                color=self.theme.copper,
                opacity=0.2,
                name='Inner Liner'
            )
            fig.add_trace(liner)

        # Configure layout
        self._apply_3d_layout(
            fig,
            title=title or f'Cooling Channels ({n_channels_to_show} of {N} shown)',
            x_label='Axial Position [mm]',
            y_label='Y [mm]',
            z_label='Z [mm]'
        )

        return fig

    def _create_channel_tube(
        self,
        x_positions: np.ndarray,
        r_bottom: np.ndarray,
        r_top: np.ndarray,
        theta_left: np.ndarray,
        theta_right: np.ndarray,
        color: str,
        name: str = '',
        show_legend: bool = True
    ) -> List[go.Mesh3d]:
        """
        Create 3D mesh for a single cooling channel tube.

        The channel is represented as a rectangular tube with 4 faces:
        - Bottom (hot wall side)
        - Top (closeout side)
        - Left rib wall
        - Right rib wall

        Args:
            x_positions: Axial coordinates
            r_bottom: Inner radius of channel (bottom of coolant)
            r_top: Outer radius of channel (top of coolant)
            theta_left: Left angular boundary
            theta_right: Right angular boundary
            color: Channel color
            name: Trace name
            show_legend: Whether to show in legend

        Returns:
            List of Mesh3d traces for the channel surfaces
        """
        traces = []
        n_x = len(x_positions)
        n_theta = 10  # Resolution across channel width

        # Bottom face (hot wall)
        theta_range = np.linspace(theta_left, theta_right, n_theta)
        if theta_range.ndim == 1:
            # theta_left/right are scalars
            X_bot = np.tile(x_positions[:, np.newaxis], (1, n_theta))
            Theta_bot = np.tile(theta_range, (n_x, 1))
            R_bot = np.tile(r_bottom[:, np.newaxis], (1, n_theta))
        else:
            # theta_left/right vary with x
            X_bot = np.tile(x_positions[:, np.newaxis], (1, n_theta))
            Theta_bot = np.array([np.linspace(theta_left[i], theta_right[i], n_theta)
                                  for i in range(n_x)])
            R_bot = np.tile(r_bottom[:, np.newaxis], (1, n_theta))

        Y_bot = R_bot * np.cos(Theta_bot)
        Z_bot = R_bot * np.sin(Theta_bot)

        traces.append(self._mesh3d_from_surface(
            X_bot, Y_bot, Z_bot,
            color=color,
            opacity=0.9,
            name=name if show_legend else ''
        ))

        # Top face (closeout side)
        R_top = np.tile(r_top[:, np.newaxis], (1, n_theta))
        Y_top = R_top * np.cos(Theta_bot)
        Z_top = R_top * np.sin(Theta_bot)

        traces.append(self._mesh3d_from_surface(
            X_bot, Y_top, Z_top,  # Same X and Theta
            color=color,
            opacity=0.7,
            name=''
        ))

        # Left wall
        n_r = 5  # Radial resolution
        r_range = np.array([np.linspace(r_bottom[i], r_top[i], n_r) for i in range(n_x)])
        X_left = np.tile(x_positions[:, np.newaxis], (1, n_r))

        if theta_left.ndim == 0 or len(np.unique(theta_left)) == 1:
            theta_l = theta_left if np.isscalar(theta_left) else theta_left[0]
            Y_left = r_range * np.cos(theta_l)
            Z_left = r_range * np.sin(theta_l)
        else:
            Y_left = r_range * np.cos(theta_left[:, np.newaxis])
            Z_left = r_range * np.sin(theta_left[:, np.newaxis])

        traces.append(self._mesh3d_from_surface(
            X_left, Y_left, Z_left,
            color=color,
            opacity=0.5,
            name=''
        ))

        # Right wall
        if theta_right.ndim == 0 or len(np.unique(theta_right)) == 1:
            theta_r = theta_right if np.isscalar(theta_right) else theta_right[0]
            Y_right = r_range * np.cos(theta_r)
            Z_right = r_range * np.sin(theta_r)
        else:
            Y_right = r_range * np.cos(theta_right[:, np.newaxis])
            Z_right = r_range * np.sin(theta_right[:, np.newaxis])

        traces.append(self._mesh3d_from_surface(
            X_left, Y_right, Z_right,  # Same X and R
            color=color,
            opacity=0.5,
            name=''
        ))

        return traces

    def _get_channel_colors(self, n_channels: int) -> List[str]:
        """Generate a list of distinct colors for channels."""
        base_colors = [
            '#3498db',  # Blue
            '#2ecc71',  # Green
            '#9b59b6',  # Purple
            '#e74c3c',  # Red
            '#f39c12',  # Orange
            '#1abc9c',  # Teal
            '#e91e63',  # Pink
            '#00bcd4',  # Cyan
        ]

        if n_channels <= len(base_colors):
            return base_colors[:n_channels]
        else:
            # Cycle through colors
            return [base_colors[i % len(base_colors)] for i in range(n_channels)]

    def render_full_engine(
        self,
        nozzle_geometry: 'NozzleGeometryData',
        channel_geometry: 'CoolingChannelGeometry',
        sector_angle: float = 270.0,
        closeout_thickness: float = 0.001,
        show_channels: bool = True,
        n_channels_to_show: int = 4,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Render complete engine assembly with nozzle, channels, and closeout.

        Creates a sectional cutaway view showing:
        - Inner liner (nozzle wall)
        - Cooling channels
        - Outer closeout/jacket

        Args:
            nozzle_geometry: NozzleGeometryData for inner contour
            channel_geometry: CoolingChannelGeometry for cooling passages
            sector_angle: Angular extent for cutaway view (degrees)
            closeout_thickness: Outer jacket thickness [m]
            show_channels: If True, render cooling channels
            n_channels_to_show: Number of channels to visualize
            title: Optional custom title

        Returns:
            Plotly Figure with full engine 3D visualization
        """
        fig = go.Figure()

        geo = channel_geometry

        # Extract and prepare geometry data
        xs = geo.x_contour * 1000  # mm
        rs_inner = geo.radius_contour * 1000
        t_wall = geo.wall_thickness * 1000
        h_ch = geo.channel_height * 1000
        rs_outer = rs_inner + t_wall + h_ch + (closeout_thickness * 1000)

        theta_end = np.radians(sector_angle)
        n_theta = 60

        # 1. Inner liner surface
        X_inner, Y_inner, Z_inner = self._create_surface_of_revolution(
            xs, rs_inner,
            n_theta=n_theta,
            theta_end=theta_end
        )

        inner_surface = self._mesh3d_from_surface(
            X_inner, Y_inner, Z_inner,
            color=self.theme.copper,
            opacity=1.0,
            name='Inner Liner (Hot Wall)'
        )
        fig.add_trace(inner_surface)

        # Add wireframe to inner surface
        wireframe = self._create_wireframe(
            X_inner, Y_inner, Z_inner,
            color='rgba(0,0,0,0.2)' if not self.dark_mode else 'rgba(255,255,255,0.2)',
            width=0.5,
            n_axial=10,
            n_circ=8
        )
        for trace in wireframe:
            fig.add_trace(trace)

        # 2. Outer closeout surface
        X_outer, Y_outer, Z_outer = self._create_surface_of_revolution(
            xs, rs_outer,
            n_theta=n_theta,
            theta_end=theta_end
        )

        outer_surface = self._mesh3d_from_surface(
            X_outer, Y_outer, Z_outer,
            color=self.theme.steel,
            opacity=0.4,
            name='Closeout Jacket'
        )
        fig.add_trace(outer_surface)

        # 3. Cooling channels (if requested)
        if show_channels and n_channels_to_show > 0:
            # Downsample for performance
            step = max(1, len(xs) // 50)
            xs_ch = xs[::step]
            r_bottom = (rs_inner + t_wall)[::step]
            r_top = (rs_inner + t_wall + h_ch)[::step]
            w_ch = geo.channel_width[::step] * 1000
            w_rib = geo.rib_width[::step] * 1000

            N = geo.number_of_channels
            pitch_angle = 2 * np.pi / N
            total_arc = w_ch + w_rib
            theta_widths = pitch_angle * (w_ch / total_arc)

            colors = self._get_channel_colors(n_channels_to_show)

            for k in range(n_channels_to_show):
                theta_center = k * pitch_angle
                theta_left = theta_center - theta_widths / 2
                theta_right = theta_center + theta_widths / 2

                # Only show channels within the sector
                if theta_center < theta_end:
                    channel_traces = self._create_channel_tube(
                        xs_ch, r_bottom, r_top, theta_left, theta_right,
                        color=colors[k],
                        name=f'Coolant Ch. {k+1}' if k == 0 else '',
                        show_legend=(k == 0)
                    )
                    for trace in channel_traces:
                        fig.add_trace(trace)

        # 4. Cut plane end caps
        self._add_cut_plane_caps(fig, xs, rs_inner, rs_outer, theta_end)

        # 5. Profile lines on cut planes
        # At theta = 0
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=rs_inner,
            z=np.zeros_like(xs),
            mode='lines',
            line=dict(color='black', width=2),
            name='Inner Profile',
            hovertemplate='X: %{x:.1f} mm<br>R: %{y:.2f} mm<extra></extra>'
        ))
        fig.add_trace(go.Scatter3d(
            x=xs,
            y=rs_outer,
            z=np.zeros_like(xs),
            mode='lines',
            line=dict(color='gray', width=2),
            name='Outer Profile',
            hovertemplate='X: %{x:.1f} mm<br>R: %{y:.2f} mm<extra></extra>'
        ))

        # Configure layout
        self._apply_3d_layout(
            fig,
            title=title or f'Engine Assembly ({sector_angle:.0f} deg Cutaway)',
            x_label='Axial Position [mm]',
            y_label='Y [mm]',
            z_label='Z [mm]'
        )

        return fig

    def _add_cut_plane_caps(
        self,
        fig: go.Figure,
        x_positions: np.ndarray,
        r_inner: np.ndarray,
        r_outer: np.ndarray,
        theta_end: float
    ) -> None:
        """Add visual caps at the cut plane edges."""
        # Radial lines at inlet and exit
        for i in [0, -1]:
            x_cap = x_positions[i]
            r_in = r_inner[i]
            r_out = r_outer[i]

            # At theta = 0
            fig.add_trace(go.Scatter3d(
                x=[x_cap, x_cap],
                y=[r_in, r_out],
                z=[0, 0],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # At theta = theta_end
            fig.add_trace(go.Scatter3d(
                x=[x_cap, x_cap],
                y=[r_in * np.cos(theta_end), r_out * np.cos(theta_end)],
                z=[r_in * np.sin(theta_end), r_out * np.sin(theta_end)],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    def render_with_temperature(
        self,
        geometry: Union['NozzleGeometryData', 'CoolingChannelGeometry'],
        temperature_data: np.ndarray,
        sector_angle: float = 360.0,
        colorbar_title: str = 'Temperature [K]',
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Render geometry with temperature color mapping.

        Creates a 3D surface colored by temperature (or heat flux, pressure, etc.)
        with an interactive colorbar.

        Args:
            geometry: NozzleGeometryData or CoolingChannelGeometry
            temperature_data: Array of values to map to colors (same length as x)
            sector_angle: Angular extent (degrees)
            colorbar_title: Title for the colorbar
            cmin, cmax: Optional color scale limits
            title: Optional plot title

        Returns:
            Plotly Figure with temperature-mapped 3D visualization
        """
        # Determine geometry type and extract profile
        if hasattr(geometry, 'x_full'):
            # NozzleGeometryData
            x_profile = geometry.x_full
            r_profile = geometry.y_full
        else:
            # CoolingChannelGeometry
            x_profile = geometry.x_contour * 1000
            r_profile = geometry.radius_contour * 1000

        # Validate data length
        if len(temperature_data) != len(x_profile):
            # Interpolate to match
            x_temp = np.linspace(0, 1, len(temperature_data))
            x_target = np.linspace(0, 1, len(x_profile))
            temperature_data = np.interp(x_target, x_temp, temperature_data)

        # Create surface of revolution
        theta_end = np.radians(sector_angle)
        n_theta = 60
        X, Y, Z = self._create_surface_of_revolution(
            x_profile, r_profile,
            n_theta=n_theta,
            theta_end=theta_end
        )

        # Create intensity grid (replicate along theta)
        intensity = np.tile(temperature_data[:, np.newaxis], (1, n_theta))

        fig = go.Figure()

        # Add temperature-mapped surface
        surface = self._mesh3d_from_surface(
            X, Y, Z,
            intensity=intensity,
            colorscale=self.theme.temperature_colorscale,
            showscale=True,
            cmin=cmin,
            cmax=cmax,
            colorbar_title=colorbar_title,
            opacity=1.0,
            name='Temperature Field'
        )
        fig.add_trace(surface)

        # Add subtle wireframe
        wireframe = self._create_wireframe(
            X, Y, Z,
            color='rgba(0,0,0,0.15)' if not self.dark_mode else 'rgba(255,255,255,0.15)',
            width=0.5,
            n_axial=12,
            n_circ=8
        )
        for trace in wireframe:
            fig.add_trace(trace)

        # Add profile line with hover data
        if sector_angle < 360:
            fig.add_trace(go.Scatter3d(
                x=x_profile,
                y=r_profile,
                z=np.zeros_like(x_profile),
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=2, color=temperature_data, colorscale=self.theme.temperature_colorscale),
                name='Profile',
                hovertemplate='X: %{x:.1f} mm<br>R: %{y:.2f} mm<br>T: %{marker.color:.0f} K<extra></extra>'
            ))

        # Configure layout
        self._apply_3d_layout(
            fig,
            title=title or 'Temperature Distribution (3D)',
            x_label='Axial Position [mm]',
            y_label='Y [mm]',
            z_label='Z [mm]'
        )

        return fig

    def render_heat_flux(
        self,
        geometry: Union['NozzleGeometryData', 'CoolingChannelGeometry'],
        heat_flux_data: np.ndarray,
        sector_angle: float = 360.0,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Render geometry with heat flux color mapping.

        Convenience method for heat flux visualization using appropriate
        colorscale and units.

        Args:
            geometry: Geometry object
            heat_flux_data: Heat flux values [W/m^2]
            sector_angle: Angular extent (degrees)
            title: Optional plot title

        Returns:
            Plotly Figure with heat flux visualization
        """
        # Convert to MW/m^2 for display
        q_mw = heat_flux_data / 1e6

        return self.render_with_temperature(
            geometry,
            q_mw,
            sector_angle=sector_angle,
            colorbar_title='Heat Flux [MW/m^2]',
            title=title or 'Heat Flux Distribution (3D)'
        )

    def _apply_3d_layout(
        self,
        fig: go.Figure,
        title: str,
        x_label: str = 'X',
        y_label: str = 'Y',
        z_label: str = 'Z'
    ) -> None:
        """Apply consistent 3D layout styling."""
        # Background colors based on theme
        if self.dark_mode:
            bg_color = self.theme.background
            paper_color = self.theme.paper_background
            grid_color = self.theme.grid_color
            font_color = '#ffffff'
        else:
            bg_color = 'rgb(240, 240, 240)'
            paper_color = self.theme.paper_background
            grid_color = self.theme.grid_color
            font_color = '#000000'

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(
                    size=self.theme.title_size,
                    family=self.theme.font_family,
                    color=font_color
                )
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(
                        text=x_label,
                        font=dict(size=self.theme.axis_title_size, color=font_color)
                    ),
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    tickfont=dict(size=self.theme.tick_size, color=font_color)
                ),
                yaxis=dict(
                    title=dict(
                        text=y_label,
                        font=dict(size=self.theme.axis_title_size, color=font_color)
                    ),
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    tickfont=dict(size=self.theme.tick_size, color=font_color)
                ),
                zaxis=dict(
                    title=dict(
                        text=z_label,
                        font=dict(size=self.theme.axis_title_size, color=font_color)
                    ),
                    backgroundcolor=bg_color,
                    gridcolor=grid_color,
                    showbackground=True,
                    tickfont=dict(size=self.theme.tick_size, color=font_color)
                ),
                aspectmode='data',  # Equal aspect ratio
                camera=self._default_camera
            ),
            paper_bgcolor=paper_color,
            margin=dict(l=10, r=10, t=60, b=10),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)' if not self.dark_mode else 'rgba(40,40,40,0.8)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(color=font_color)
            )
        )

    def set_camera(
        self,
        fig: go.Figure,
        elevation: float = 30.0,
        azimuth: float = -60.0,
        distance: float = 2.0
    ) -> go.Figure:
        """
        Set camera position for the 3D view.

        Args:
            fig: Plotly Figure to modify
            elevation: Elevation angle in degrees (0 = side view, 90 = top view)
            azimuth: Azimuth angle in degrees
            distance: Camera distance from origin

        Returns:
            Modified figure
        """
        # Convert spherical to Cartesian
        elev_rad = np.radians(elevation)
        azim_rad = np.radians(azimuth)

        eye = dict(
            x=distance * np.cos(elev_rad) * np.cos(azim_rad),
            y=distance * np.cos(elev_rad) * np.sin(azim_rad),
            z=distance * np.sin(elev_rad)
        )

        fig.update_layout(
            scene_camera=dict(
                eye=eye,
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        )

        return fig

    def to_html(
        self,
        fig: go.Figure,
        include_plotlyjs: str = 'cdn',
        full_html: bool = True,
        include_mathjax: bool = False
    ) -> str:
        """
        Export figure to standalone HTML.

        Args:
            fig: Plotly Figure to export
            include_plotlyjs: How to include Plotly.js:
                - 'cdn': Load from CDN (smallest file, requires internet)
                - True: Embed full Plotly.js (larger file, works offline)
                - False: Don't include (for embedding in page with Plotly.js)
                - str path: Load from custom path
            full_html: If True, include <html>, <head>, <body> tags
            include_mathjax: If True, include MathJax for LaTeX rendering

        Returns:
            HTML string
        """
        return fig.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html,
            include_mathjax='cdn' if include_mathjax else False
        )

    def save_html(
        self,
        fig: go.Figure,
        filepath: str,
        **kwargs
    ) -> None:
        """
        Save figure to HTML file.

        Args:
            fig: Plotly Figure to save
            filepath: Output file path
            **kwargs: Additional arguments passed to to_html()
        """
        html_content = self.to_html(fig, **kwargs)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def show(self, fig: go.Figure) -> None:
        """Display figure interactively."""
        fig.show()


# Convenience functions for quick visualization

def quick_nozzle_3d(
    nozzle_geometry: 'NozzleGeometryData',
    dark_mode: bool = False,
    **kwargs
) -> go.Figure:
    """
    Quick function to visualize nozzle geometry in 3D.

    Args:
        nozzle_geometry: NozzleGeometryData object
        dark_mode: Use dark theme
        **kwargs: Additional arguments passed to render_nozzle()

    Returns:
        Plotly Figure
    """
    viewer = Engine3DViewer(dark_mode=dark_mode)
    return viewer.render_nozzle(nozzle_geometry, **kwargs)


def quick_channels_3d(
    channel_geometry: 'CoolingChannelGeometry',
    n_channels: int = 4,
    dark_mode: bool = False,
    **kwargs
) -> go.Figure:
    """
    Quick function to visualize cooling channels in 3D.

    Args:
        channel_geometry: CoolingChannelGeometry object
        n_channels: Number of channels to show
        dark_mode: Use dark theme
        **kwargs: Additional arguments passed to render_channels()

    Returns:
        Plotly Figure
    """
    viewer = Engine3DViewer(dark_mode=dark_mode)
    return viewer.render_channels(channel_geometry, n_channels_to_show=n_channels, **kwargs)


def quick_engine_3d(
    nozzle_geometry: 'NozzleGeometryData',
    channel_geometry: 'CoolingChannelGeometry',
    dark_mode: bool = False,
    **kwargs
) -> go.Figure:
    """
    Quick function to visualize full engine assembly in 3D.

    Args:
        nozzle_geometry: NozzleGeometryData object
        channel_geometry: CoolingChannelGeometry object
        dark_mode: Use dark theme
        **kwargs: Additional arguments passed to render_full_engine()

    Returns:
        Plotly Figure
    """
    viewer = Engine3DViewer(dark_mode=dark_mode)
    return viewer.render_full_engine(nozzle_geometry, channel_geometry, **kwargs)
