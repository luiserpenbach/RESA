"""3D Contour Design Page for RESA UI."""
import streamlit as st
import numpy as np


def render_contour_page():
    """Render the 3D contour design page."""
    st.title("3D Contour Design")

    tab1, tab2, tab3 = st.tabs(["Nozzle Design", "Cooling Channels", "Export"])

    with tab1:
        render_nozzle_tab()

    with tab2:
        render_channels_tab()

    with tab3:
        render_export_tab()


def render_nozzle_tab():
    """Render nozzle contour design."""
    st.subheader("Bell Nozzle Contour")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Throat & Chamber")
        R_t = st.number_input(
            "Throat Radius [mm]",
            5.0, 100.0, 15.0,
            key="contour_rt"
        )
        CR = st.number_input(
            "Contraction Ratio (Ac/At)",
            2.0, 10.0, 5.0,
            key="contour_cr"
        )
        L_star = st.number_input(
            "L* [m]",
            0.5, 2.0, 1.0,
            key="contour_lstar"
        )

    with col2:
        st.markdown("### Nozzle Expansion")
        ER = st.number_input(
            "Expansion Ratio (Ae/At)",
            2.0, 50.0, 10.0,
            key="contour_er"
        )
        theta_n = st.slider(
            "Initial Expansion Angle [deg]",
            20.0, 45.0, 32.0,
            key="contour_theta_n"
        )
        theta_e = st.slider(
            "Exit Angle [deg]",
            5.0, 15.0, 8.0,
            key="contour_theta_e"
        )
        L_pct = st.slider(
            "Bell Length [% of 15deg cone]",
            60, 100, 80,
            key="contour_l_pct"
        )

    st.markdown("### Mesh Resolution")
    col3, col4 = st.columns(2)
    with col3:
        n_axial = st.slider("Axial Points", 50, 500, 200)
    with col4:
        n_radial = st.slider("Circumferential Points", 16, 128, 64)

    if st.button("Generate Nozzle", type="primary"):
        with st.spinner("Generating 3D geometry..."):
            try:
                from resa.addons.contour import (
                    NozzleParameters,
                    Nozzle3DGenerator
                )

                params = NozzleParameters(
                    R_t=R_t / 1000,  # mm -> m
                    CR=CR,
                    L_star=L_star,
                    ER=ER,
                    theta_n=theta_n,
                    theta_e=theta_e,
                    L_percent=L_pct
                )

                generator = Nozzle3DGenerator(params)
                st.session_state.nozzle_generator = generator
                st.session_state.nozzle_params = params

                st.success("Nozzle geometry generated!")

                # Display geometry info
                geom = generator.geometry
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chamber Dia", f"{geom.R_c*2000:.1f} mm")
                with col2:
                    st.metric("Throat Dia", f"{params.R_t*2000:.1f} mm")
                with col3:
                    st.metric("Exit Dia", f"{geom.R_e*2000:.1f} mm")

                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("Chamber Length", f"{geom.L_c*1000:.1f} mm")
                with col5:
                    st.metric("Conv Length", f"{geom.L_conv*1000:.1f} mm")
                with col6:
                    st.metric("Nozzle Length", f"{geom.L_div*1000:.1f} mm")

                # 3D Preview
                render_3d_preview(generator)

            except ImportError as e:
                st.error(f"Import error: {e}")
            except Exception as e:
                st.error(f"Generation failed: {e}")


def render_channels_tab():
    """Render cooling channel design."""
    st.subheader("Regenerative Cooling Channels")

    if not st.session_state.get('nozzle_generator'):
        st.warning("Generate a nozzle first in the Nozzle Design tab.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Wall Properties")
        wall_thickness = st.number_input(
            "Inner Wall Thickness [mm]",
            0.5, 5.0, 1.0,
            key="ch_wall_thick"
        )

        st.markdown("### Channel Dimensions (at throat)")
        ch_width = st.number_input(
            "Channel Width [mm]",
            1.0, 10.0, 3.0,
            key="ch_width"
        )
        ch_height = st.number_input(
            "Channel Height [mm]",
            1.0, 15.0, 4.0,
            key="ch_height"
        )
        ch_rib = st.number_input(
            "Rib Width [mm]",
            1.0, 5.0, 1.5,
            key="ch_rib"
        )

    with col2:
        st.markdown("### Channel Type")
        ch_type = st.radio(
            "Channel Configuration",
            ["straight", "helical"],
            format_func=lambda x: "Straight (Axial)" if x == "straight" else "Helical (Spiral)",
            key="ch_type"
        )

        if ch_type == "helical":
            helix_angle = st.slider(
                "Helix Angle at Throat [deg]",
                10.0, 45.0, 20.0,
                key="ch_helix"
            )
        else:
            helix_angle = 0.0

        st.markdown("### Number of Channels")
        n_channels = st.number_input(
            "Channel Count",
            8, 128, 32,
            key="ch_count"
        )

    if st.button("Generate Channels", type="primary"):
        with st.spinner("Generating channel geometry..."):
            try:
                from resa.addons.contour import (
                    CoolingChannelParameters,
                    CoolingChannel3DGenerator
                )

                ch_params = CoolingChannelParameters(
                    inner_wall_thickness=wall_thickness / 1000,
                    channel_width_throat=ch_width / 1000,
                    channel_height_throat=ch_height / 1000,
                    rib_width_throat=ch_rib / 1000,
                    n_channels=n_channels,
                    channel_type=ch_type,
                    helix_angle_throat=helix_angle if ch_type == "helical" else 0.0
                )

                nozzle_params = st.session_state.nozzle_params
                channel_gen = CoolingChannel3DGenerator(nozzle_params, ch_params)

                st.session_state.channel_generator = channel_gen
                st.session_state.channel_params = ch_params

                st.success(f"Generated {n_channels} cooling channels!")

            except Exception as e:
                st.error(f"Channel generation failed: {e}")


def render_export_tab():
    """Render geometry export options."""
    st.subheader("Export Geometry")

    has_nozzle = st.session_state.get('nozzle_generator') is not None
    has_channels = st.session_state.get('channel_generator') is not None

    if not has_nozzle:
        st.warning("No geometry to export. Generate a nozzle first.")
        return

    st.markdown("### STL Export")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Nozzle Geometry**")
        export_inner = st.checkbox("Inner Surface (Flow Path)", value=True)
        export_outer = st.checkbox("Outer Surface", value=True)

    with col2:
        st.markdown("**Cooling Channels**")
        if has_channels:
            export_channels = st.checkbox("Export Channels", value=True)
            channel_format = st.selectbox(
                "Format",
                ["single", "individual"],
                format_func=lambda x: "Single Combined" if x == "single" else "Individual Files"
            )
        else:
            st.info("Generate channels first to enable export.")
            export_channels = False

    st.markdown("### Output Settings")
    output_dir = st.text_input(
        "Output Directory",
        value="./output/geometry",
        key="export_dir"
    )
    file_prefix = st.text_input(
        "File Prefix",
        value="engine",
        key="export_prefix"
    )

    stl_format = st.radio(
        "STL Format",
        ["binary", "ascii"],
        format_func=lambda x: "Binary (smaller file)" if x == "binary" else "ASCII (human readable)"
    )

    if st.button("Export STL Files", type="primary"):
        with st.spinner("Exporting..."):
            try:
                from pathlib import Path
                from resa.addons.contour import export_stl_binary, export_stl_ascii

                export_func = export_stl_binary if stl_format == "binary" else export_stl_ascii
                out_path = Path(output_dir)
                out_path.mkdir(parents=True, exist_ok=True)

                exported = []

                nozzle_gen = st.session_state.nozzle_generator

                if export_inner:
                    fname = out_path / f"{file_prefix}_inner.stl"
                    nozzle_gen.export_inner_stl(str(fname))
                    exported.append(fname.name)

                if export_outer:
                    fname = out_path / f"{file_prefix}_outer.stl"
                    nozzle_gen.export_outer_stl(str(fname))
                    exported.append(fname.name)

                if export_channels and has_channels:
                    channel_gen = st.session_state.channel_generator
                    n_ch = st.session_state.channel_params.n_channels

                    if channel_format == "single":
                        fname = out_path / f"{file_prefix}_channels.stl"
                        channel_gen.export_all_channels_stl(str(fname))
                        exported.append(fname.name)
                    else:
                        for i in range(n_ch):
                            fname = out_path / f"{file_prefix}_channel_{i:02d}.stl"
                            channel_gen.export_channel_stl(str(fname), i)
                            exported.append(fname.name)

                st.success(f"Exported {len(exported)} file(s) to {output_dir}")

                with st.expander("Exported Files"):
                    for f in exported:
                        st.text(f"- {f}")

            except Exception as e:
                st.error(f"Export failed: {e}")

    st.markdown("---")
    st.markdown("### JSON Export")
    st.markdown("Export geometry parameters for documentation or data exchange.")

    if st.button("Export JSON"):
        try:
            from pathlib import Path
            from resa.addons.contour import export_geometry_json

            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            nozzle_params = st.session_state.nozzle_params
            data = {"nozzle": nozzle_params.__dict__}

            if has_channels:
                data["channels"] = st.session_state.channel_params.__dict__

            fname = out_path / f"{file_prefix}_geometry.json"
            export_geometry_json(data, str(fname))

            st.success(f"Exported to {fname}")

        except Exception as e:
            st.error(f"JSON export failed: {e}")


def render_3d_preview(generator):
    """Render 3D preview using Plotly."""
    st.markdown("### 3D Preview")

    try:
        import plotly.graph_objects as go

        geom = generator.geometry

        # Generate contour points
        x = np.linspace(0, geom.L_total, 200)
        r = np.array([generator._radius_at_x(xi) for xi in x])

        # Create surface of revolution
        theta = np.linspace(0, 2 * np.pi, 64)
        X, Theta = np.meshgrid(x, theta)
        R = np.zeros_like(X)
        for i, xi in enumerate(x):
            R[:, i] = generator._radius_at_x(xi)

        Y = R * np.cos(Theta)
        Z = R * np.sin(Theta)

        fig = go.Figure()

        # Add surface
        fig.add_trace(go.Surface(
            x=X * 1000,  # Convert to mm
            y=Y * 1000,
            z=Z * 1000,
            colorscale='Viridis',
            showscale=False,
            opacity=0.9,
            name='Nozzle'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X [mm]",
                yaxis_title="Y [mm]",
                zaxis_title="Z [mm]",
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"3D preview unavailable: {e}")

        # Fallback to 2D contour
        st.markdown("**2D Contour Profile**")
        x = np.linspace(0, generator.geometry.L_total, 200)
        r = np.array([generator._radius_at_x(xi) for xi in x])

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x * 1000, y=r * 1000,
            mode='lines', name='Upper',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x * 1000, y=-r * 1000,
            mode='lines', name='Lower',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.update_layout(
            xaxis_title="X [mm]",
            yaxis_title="R [mm]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
