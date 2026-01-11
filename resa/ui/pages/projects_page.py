"""Project Management Page for RESA UI."""
import streamlit as st
from datetime import datetime


def render_projects_page():
    """Render the project management page."""
    st.title("Project Management")

    tab1, tab2, tab3, tab4 = st.tabs(["Projects", "Version History", "Compare", "Settings"])

    with tab1:
        render_projects_tab()

    with tab2:
        render_versions_tab()

    with tab3:
        render_compare_tab()

    with tab4:
        render_settings_tab_internal()


def render_projects_tab():
    """Render project selection and creation."""
    st.subheader("Projects")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Recent Projects")

        # List existing projects
        projects = get_projects_list()

        if not projects:
            st.info("No projects found. Create a new project to get started.")
        else:
            for proj in projects:
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 2, 1])
                    with col_a:
                        st.markdown(f"**{proj['name']}**")
                        st.caption(proj.get('description', 'No description'))
                    with col_b:
                        st.caption(f"Modified: {proj.get('modified', 'Unknown')}")
                        st.caption(f"Versions: {proj.get('n_versions', 0)}")
                    with col_c:
                        if st.button("Open", key=f"open_{proj['name']}"):
                            st.session_state.current_project = proj['name']
                            st.session_state.project_dir = proj['path']
                            st.success(f"Opened project: {proj['name']}")

                    st.markdown("---")

    with col2:
        st.markdown("### New Project")

        project_name = st.text_input("Project Name", key="new_proj_name")
        project_desc = st.text_area("Description", key="new_proj_desc", height=100)
        project_dir = st.text_input(
            "Project Directory",
            value="./projects",
            key="new_proj_dir"
        )

        if st.button("Create Project", type="primary"):
            if not project_name:
                st.error("Project name is required")
            else:
                try:
                    from pathlib import Path
                    proj_path = Path(project_dir) / project_name
                    proj_path.mkdir(parents=True, exist_ok=True)

                    # Create project metadata
                    import json
                    meta = {
                        "name": project_name,
                        "description": project_desc,
                        "created": datetime.now().isoformat(),
                        "author": st.session_state.get('author_name', 'Unknown')
                    }
                    with open(proj_path / "project.json", 'w') as f:
                        json.dump(meta, f, indent=2)

                    st.session_state.current_project = project_name
                    st.session_state.project_dir = str(proj_path)

                    st.success(f"Created project: {project_name}")

                except Exception as e:
                    st.error(f"Failed to create project: {e}")

        st.markdown("---")

        # Current project info
        if st.session_state.get('current_project'):
            st.markdown("### Current Project")
            st.info(st.session_state.current_project)


def render_versions_tab():
    """Render version history for current project."""
    st.subheader("Version History")

    if not st.session_state.get('project_dir'):
        st.warning("Open a project first.")
        return

    try:
        from resa.projects.version_control import ProjectVersionControl

        vc = ProjectVersionControl(st.session_state.project_dir)
        versions = vc.list_versions()

        if not versions:
            st.info("No versions saved yet. Save a design to create the first version.")
            return

        # Current version
        current = vc.get_current()
        if current:
            st.markdown(f"**Current Version:** `{current.version_id[:8]}...` - {current.description}")

        st.markdown("---")

        # Version list
        for v in versions:
            is_current = current and v.version_id == current.version_id
            marker = " (current)" if is_current else ""

            with st.expander(
                f"{v.timestamp.strftime('%Y-%m-%d %H:%M')} - {v.description}{marker}",
                expanded=is_current
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Metadata**")
                    st.text(f"ID: {v.version_id[:12]}...")
                    st.text(f"Author: {v.author}")
                    if v.tags:
                        st.text(f"Tags: {', '.join(v.tags)}")

                with col2:
                    st.markdown("**Key Metrics**")
                    metrics = v.metrics
                    if metrics:
                        st.text(f"Thrust (vac): {metrics.get('thrust_vac', 0):.0f} N")
                        st.text(f"Isp (vac): {metrics.get('isp_vac', 0):.1f} s")
                        st.text(f"Pc: {metrics.get('pc_bar', 0):.1f} bar")

                # Actions
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("Load", key=f"load_{v.version_id[:8]}"):
                        config, result = vc.load_version(v.version_id)
                        st.session_state.engine_config = config
                        st.session_state.loaded_result = result
                        vc.checkout(v.version_id)
                        st.success("Version loaded!")

                with col4:
                    tag_name = st.text_input("Tag", key=f"tag_{v.version_id[:8]}", label_visibility="collapsed", placeholder="Tag name")
                    if st.button("Add Tag", key=f"addtag_{v.version_id[:8]}"):
                        if tag_name:
                            vc.tag_version(v.version_id, tag_name)
                            st.success(f"Added tag: {tag_name}")

                with col5:
                    if st.button("Compare", key=f"compare_{v.version_id[:8]}"):
                        st.session_state.compare_version = v.version_id

    except ImportError:
        st.error("Version control module not available.")
    except Exception as e:
        st.error(f"Error loading versions: {e}")


def render_compare_tab():
    """Render version comparison."""
    st.subheader("Compare Versions")

    if not st.session_state.get('project_dir'):
        st.warning("Open a project first.")
        return

    try:
        from resa.projects.version_control import ProjectVersionControl

        vc = ProjectVersionControl(st.session_state.project_dir)
        versions = vc.list_versions()

        if len(versions) < 2:
            st.info("Need at least 2 versions to compare.")
            return

        version_options = {
            f"{v.timestamp.strftime('%Y-%m-%d %H:%M')} - {v.description[:30]}": v.version_id
            for v in versions
        }

        col1, col2 = st.columns(2)

        with col1:
            v1_label = st.selectbox(
                "Base Version",
                list(version_options.keys()),
                key="compare_v1"
            )

        with col2:
            v2_label = st.selectbox(
                "Compare To",
                list(version_options.keys()),
                index=1 if len(version_options) > 1 else 0,
                key="compare_v2"
            )

        if st.button("Compare", type="primary"):
            v1_id = version_options[v1_label]
            v2_id = version_options[v2_label]

            diff = vc.diff_versions(v1_id, v2_id)

            st.markdown("---")

            # Config changes
            st.markdown("### Configuration Changes")
            if diff['config_changes']:
                for key, change in diff['config_changes'].items():
                    col_a, col_b, col_c = st.columns([2, 1, 1])
                    with col_a:
                        st.text(key)
                    with col_b:
                        st.text(f"{change['from']}")
                    with col_c:
                        st.text(f"{change['to']}")
            else:
                st.info("No configuration changes.")

            # Metric changes
            st.markdown("### Performance Changes")
            if diff['metric_changes']:
                import plotly.graph_objects as go

                metrics = list(diff['metric_changes'].keys())
                pct_changes = [diff['metric_changes'][m]['pct_change'] for m in metrics]

                colors = ['green' if p > 0 else 'red' for p in pct_changes]

                fig = go.Figure(go.Bar(
                    x=metrics,
                    y=pct_changes,
                    marker_color=colors,
                    text=[f"{p:+.1f}%" for p in pct_changes],
                    textposition='outside'
                ))

                fig.update_layout(
                    title="Metric Changes (%)",
                    yaxis_title="Change [%]",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Detailed table
                with st.expander("Detailed Changes"):
                    for metric, change in diff['metric_changes'].items():
                        st.markdown(f"**{metric}**")
                        st.text(f"  {change['from']:.3f} -> {change['to']:.3f} ({change['pct_change']:+.1f}%)")
            else:
                st.info("No metric changes.")

    except Exception as e:
        st.error(f"Comparison failed: {e}")


def render_settings_tab_internal():
    """Render project settings."""
    st.subheader("Output & Settings")

    st.markdown("### Output Directory")

    output_dir = st.text_input(
        "Default Output Directory",
        value=st.session_state.get('output_dir', './output'),
        key="output_dir_setting"
    )

    if st.button("Set Output Directory"):
        from pathlib import Path
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            st.session_state.output_dir = output_dir
            st.success(f"Output directory set to: {output_dir}")
        except Exception as e:
            st.error(f"Invalid directory: {e}")

    st.markdown("---")
    st.markdown("### User Settings")

    author_name = st.text_input(
        "Author Name",
        value=st.session_state.get('author_name', ''),
        key="author_setting"
    )

    if st.button("Save Author"):
        st.session_state.author_name = author_name
        st.success("Author name saved!")

    st.markdown("---")
    st.markdown("### Report Templates")

    template = st.selectbox(
        "Report Template",
        ["Standard", "Detailed", "Summary", "Custom"],
        key="report_template"
    )

    if template == "Custom":
        custom_template = st.text_area(
            "Custom Template Path",
            placeholder="Path to custom HTML template",
            key="custom_template"
        )

    st.markdown("---")
    st.markdown("### Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export All Data"):
            st.info("Would export all project data to JSON/ZIP")

    with col2:
        uploaded = st.file_uploader("Import Project", type=['zip', 'json'])
        if uploaded:
            st.info(f"Would import: {uploaded.name}")


def get_projects_list():
    """Get list of existing projects."""
    from pathlib import Path

    projects = []
    projects_dir = Path("./projects")

    if not projects_dir.exists():
        return projects

    for proj_dir in projects_dir.iterdir():
        if proj_dir.is_dir():
            meta_file = proj_dir / "project.json"
            if meta_file.exists():
                try:
                    import json
                    with open(meta_file) as f:
                        meta = json.load(f)

                    # Count versions
                    versions_dir = proj_dir / "versions"
                    n_versions = 0
                    if versions_dir.exists():
                        n_versions = len(list(versions_dir.glob("*.json"))) - 1  # Exclude index

                    projects.append({
                        "name": meta.get("name", proj_dir.name),
                        "description": meta.get("description", ""),
                        "path": str(proj_dir),
                        "modified": meta.get("created", "Unknown")[:10],
                        "n_versions": max(0, n_versions)
                    })
                except Exception:
                    # Invalid project, skip
                    pass

    return sorted(projects, key=lambda x: x.get('modified', ''), reverse=True)
