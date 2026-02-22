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
            ["straight"],
            format_func=lambda x: "Straight (Axial)",
            key="ch_type"
        )

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

    # ── Visualization ─────────────────────────────────────────────────
    if st.session_state.get("channel_generator"):
        st.divider()
        _render_channel_views()


def _render_channel_views():
    """Render cross-section and 3D views of cooling channels."""
    import plotly.graph_objects as go
    from resa.addons.contour import (
        compute_channel_geometry_at_x,
        generate_2d_contour,
    )

    channel_gen = st.session_state.channel_generator
    ch_params = st.session_state.channel_params
    nozzle_params = st.session_state.nozzle_params
    geom = channel_gen.geometry

    x_profile, r_profile = generate_2d_contour(nozzle_params, 200)

    view_tab1, view_tab2 = st.tabs(["Cross-Section View", "3D Channel View"])

    # ── Cross-section at user-selected axial station ──────────────────
    with view_tab1:
        st.markdown("### Cross-Section at Axial Station")
        x_throat = geom.L_c + geom.L_conv
        x_min_mm = float(x_profile[0]) * 1000
        x_max_mm = float(x_profile[-1]) * 1000
        x_sel_mm = st.slider(
            "Axial position [mm]",
            x_min_mm, x_max_mm,
            float(x_throat * 1000),
            step=0.5,
            key="ch_xsec_pos",
        )
        x_sel = x_sel_mm / 1000.0
        r_inner = float(np.interp(x_sel, x_profile, r_profile))

        ch_geom = compute_channel_geometry_at_x(
            x_sel, r_inner, nozzle_params, ch_params, geom
        )
        n_ch = ch_geom["n_channels"]
        w_ch = ch_geom["channel_width"]
        h_ch = ch_geom["channel_height"]
        w_rib = ch_geom["rib_width"]
        r_bottom = r_inner + ch_params.inner_wall_thickness
        r_top = r_bottom + h_ch
        r_outer = r_top + ch_params.outer_wall_thickness

        # Build figure
        fig = go.Figure()
        theta_full = np.linspace(0, 2 * np.pi, 256)

        # Inner wall circle
        fig.add_trace(go.Scatter(
            x=r_inner * np.cos(theta_full) * 1000,
            y=r_inner * np.sin(theta_full) * 1000,
            mode="lines", name="Inner wall (hot-gas)",
            line=dict(color="#ff5722", width=1.5),
        ))
        # Channel bottom circle
        fig.add_trace(go.Scatter(
            x=r_bottom * np.cos(theta_full) * 1000,
            y=r_bottom * np.sin(theta_full) * 1000,
            mode="lines", name="Channel floor",
            line=dict(color="#ff9d3a", width=1, dash="dot"),
        ))
        # Channel top circle
        fig.add_trace(go.Scatter(
            x=r_top * np.cos(theta_full) * 1000,
            y=r_top * np.sin(theta_full) * 1000,
            mode="lines", name="Channel ceiling",
            line=dict(color="#4a9eff", width=1, dash="dot"),
        ))
        # Outer wall circle
        fig.add_trace(go.Scatter(
            x=r_outer * np.cos(theta_full) * 1000,
            y=r_outer * np.sin(theta_full) * 1000,
            mode="lines", name="Outer wall",
            line=dict(color="#c084fc", width=1.5),
        ))

        # Draw channel slots (show all, filled)
        theta_pitch = ch_geom["theta_pitch"]
        theta_half_ch = ch_geom["theta_channel"] / 2
        for i in range(n_ch):
            tc = i * theta_pitch
            t0 = tc - theta_half_ch
            t1 = tc + theta_half_ch

            # Channel rectangle (bottom-left → bottom-right → top-right → top-left)
            arc_b = np.linspace(t0, t1, 12)
            arc_t = np.linspace(t1, t0, 12)
            xs = np.concatenate([
                r_bottom * np.cos(arc_b),
                r_top * np.cos(arc_t),
                [r_bottom * np.cos(t0)],
            ]) * 1000
            ys = np.concatenate([
                r_bottom * np.sin(arc_b),
                r_top * np.sin(arc_t),
                [r_bottom * np.sin(t0)],
            ]) * 1000
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                fill="toself", fillcolor="rgba(74,158,255,0.15)",
                line=dict(color="#4a9eff", width=0.8),
                showlegend=(i == 0), name="Coolant channel",
            ))

        fig.update_layout(
            title=f"Cross-Section at x = {x_sel_mm:.1f} mm  (r = {r_inner*1000:.2f} mm)",
            xaxis_title="Y [mm]", yaxis_title="Z [mm]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            height=550, hovermode="closest",
        )
        try:
            from resa.visualization.themes import DarkTheme
            DarkTheme().apply_to_figure(fig)
        except ImportError:
            pass
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Channels", n_ch)
        with c2:
            st.metric("Width", f"{w_ch*1000:.2f} mm")
        with c3:
            st.metric("Height", f"{h_ch*1000:.2f} mm")
        with c4:
            st.metric("Rib", f"{w_rib*1000:.2f} mm")

    # ── 3D channel view ───────────────────────────────────────────────
    with view_tab2:
        st.markdown("### 3D Channel Visualization")
        n_show = st.slider(
            "Channels to display",
            1, min(n_ch, 16), min(n_ch, 8),
            key="ch_3d_nshow",
            help="Limit displayed channels to keep the view responsive",
        )

        with st.spinner("Building 3D view..."):
            fig3d = go.Figure()

            # Semi-transparent nozzle inner surface
            from resa.addons.contour import generate_3d_surface_revolution
            n_ax_3d = 100
            x_p, r_p = generate_2d_contour(nozzle_params, n_ax_3d)
            X_s, Y_s, Z_s = generate_3d_surface_revolution(x_p, r_p, 48)
            fig3d.add_trace(go.Surface(
                x=X_s * 1000, y=Y_s * 1000, z=Z_s * 1000,
                colorscale=[[0, "rgba(100,130,160,0.25)"],
                            [1, "rgba(100,130,160,0.25)"]],
                showscale=False, opacity=0.3, name="Nozzle",
                hoverinfo="skip",
            ))

            # Pre-compute channel geometry at each axial station (shared across channels)
            r_bottom_prof = r_p + ch_params.inner_wall_thickness
            r_top_prof = np.zeros_like(r_p)
            half_angle_arr = np.zeros(len(x_p))
            for k, (xv, rv) in enumerate(zip(x_p, r_p)):
                cg = compute_channel_geometry_at_x(
                    xv, rv, nozzle_params, ch_params, geom
                )
                r_top_prof[k] = rv + ch_params.inner_wall_thickness + cg["channel_height"]
                half_angle_arr[k] = cg["theta_channel"] / 2

            step = max(1, n_ch // n_show)
            colours = [
                "#4a9eff", "#2ecc71", "#ff6b6b", "#c084fc",
                "#1abc9c", "#ff9d3a", "#f472b6", "#a78bfa",
                "#00e5ff", "#ffeb3b", "#e8f4fd", "#ff5722",
                "#5dade2", "#e89c4f", "#c0c8d4", "#f0c040",
            ]

            for ci in range(0, n_ch, step):
                theta_c_arr = np.full(len(x_p), ci * 2 * np.pi / n_ch)

                # Build bottom and top surfaces for this channel
                n_w = 6  # angular resolution within channel width
                th_pts = np.linspace(-1, 1, n_w)  # normalised

                clr = colours[ci % len(colours)]
                cscale = [[0, clr], [1, clr]]

                # Bottom and top surfaces
                for surface_label, r_surf in [("bottom", r_bottom_prof), ("top", r_top_prof)]:
                    Xm = np.zeros((len(x_p), n_w))
                    Ym = np.zeros((len(x_p), n_w))
                    Zm = np.zeros((len(x_p), n_w))
                    for k in range(len(x_p)):
                        for jj, tp in enumerate(th_pts):
                            ang = theta_c_arr[k] + tp * half_angle_arr[k]
                            Xm[k, jj] = x_p[k] * 1000
                            Ym[k, jj] = r_surf[k] * np.cos(ang) * 1000
                            Zm[k, jj] = r_surf[k] * np.sin(ang) * 1000

                    fig3d.add_trace(go.Surface(
                        x=Xm, y=Ym, z=Zm,
                        colorscale=cscale,
                        showscale=False, opacity=0.85,
                        showlegend=(surface_label == "bottom"),
                        name=f"Ch {ci}" if surface_label == "bottom" else None,
                        hoverinfo="skip",
                    ))

                # Side walls (left and right edges)
                for side in [-1, 1]:
                    Xs = np.zeros((len(x_p), 2))
                    Ys = np.zeros((len(x_p), 2))
                    Zs = np.zeros((len(x_p), 2))
                    for k in range(len(x_p)):
                        ang = theta_c_arr[k] + side * half_angle_arr[k]
                        Xs[k, 0] = x_p[k] * 1000
                        Ys[k, 0] = r_bottom_prof[k] * np.cos(ang) * 1000
                        Zs[k, 0] = r_bottom_prof[k] * np.sin(ang) * 1000
                        Xs[k, 1] = x_p[k] * 1000
                        Ys[k, 1] = r_top_prof[k] * np.cos(ang) * 1000
                        Zs[k, 1] = r_top_prof[k] * np.sin(ang) * 1000
                    fig3d.add_trace(go.Surface(
                        x=Xs, y=Ys, z=Zs,
                        colorscale=cscale,
                        showscale=False, opacity=0.85,
                        showlegend=False, hoverinfo="skip",
                    ))

            fig3d.update_layout(
                scene=dict(
                    xaxis_title="X [mm]",
                    yaxis_title="Y [mm]",
                    zaxis_title="Z [mm]",
                    aspectmode="data",
                    camera=dict(eye=dict(x=1.5, y=1.0, z=0.6)),
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                height=560,
            )
            try:
                from resa.visualization.themes import DarkTheme
                DarkTheme().apply_to_figure(fig3d)
            except ImportError:
                pass
            st.plotly_chart(fig3d, use_container_width=True, theme=None)


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
        from resa.addons.contour import generate_3d_surface_revolution

        # Generate 2D contour via public API
        x_profile, r_profile = generator.get_2d_contour(200)

        # Create surface of revolution
        X, Y, Z = generate_3d_surface_revolution(x_profile, r_profile, 64)

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

        try:
            from resa.visualization.themes import DarkTheme
            DarkTheme().apply_to_figure(fig)
        except ImportError:
            pass
        st.plotly_chart(fig, use_container_width=True, theme=None)

    except Exception as e:
        st.warning(f"3D preview unavailable: {e}")

        # Fallback to 2D contour
        st.markdown("**2D Contour Profile**")
        x, r = generator.get_2d_contour(200)

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x * 1000, y=r * 1000,
            mode='lines', name='Upper',
            line=dict(color='#4a9eff', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x * 1000, y=-r * 1000,
            mode='lines', name='Lower',
            line=dict(color='#4a9eff', width=2)
        ))

        fig.update_layout(
            xaxis_title="X [mm]",
            yaxis_title="R [mm]",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False,
            height=300
        )
        try:
            from resa.visualization.themes import DarkTheme
            DarkTheme().apply_to_figure(fig)
        except ImportError:
            pass
        st.plotly_chart(fig, use_container_width=True, theme=None)
