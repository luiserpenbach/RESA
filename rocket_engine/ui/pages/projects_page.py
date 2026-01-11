"""
Projects Page - Project management and file handling
"""
import streamlit as st
from pathlib import Path


def render_projects_page():
    """Render the project management page."""
    st.title("📁 Project Manager")
    
    st.markdown("""
    Manage engine design projects, save/load configurations, and export results.
    """)
    
    tab_current, tab_save, tab_load, tab_export = st.tabs([
        "📋 Current Project",
        "💾 Save",
        "📂 Load",
        "📤 Export"
    ])
    
    with tab_current:
        if st.session_state.get('engine_config'):
            cfg = st.session_state.engine_config
            
            st.markdown(f"### {cfg.engine_name}")
            st.markdown(f"**Designer:** {cfg.designer or 'Not specified'}")
            st.markdown(f"**Version:** {cfg.version}")
            
            st.markdown("---")
            st.markdown("#### Configuration Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Propulsion**")
                st.text(f"Fuel: {cfg.fuel}")
                st.text(f"Oxidizer: {cfg.oxidizer}")
                st.text(f"Thrust: {cfg.thrust_n:.0f} N")
                st.text(f"Pc: {cfg.pc_bar:.1f} bar")
                st.text(f"MR: {cfg.mr:.2f}")
            
            with col2:
                st.markdown("**Cooling**")
                st.text(f"Coolant: {cfg.coolant_name.split('::')[-1]}")
                st.text(f"Inlet P: {cfg.coolant_p_in_bar:.1f} bar")
                st.text(f"Inlet T: {cfg.coolant_t_in_k:.1f} K")
            
            if st.session_state.get('design_result'):
                st.markdown("---")
                st.markdown("#### Last Analysis Result")
                res = st.session_state.design_result
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Thrust", f"{res.thrust_sea:.0f} N")
                with col2:
                    st.metric("Isp", f"{res.isp_sea:.1f} s")
                with col3:
                    if res.cooling:
                        st.metric("Max T_wall", f"{res.cooling.max_wall_temp:.0f} K")
        else:
            st.info("No project loaded. Create a new design or load an existing configuration.")
    
    with tab_save:
        st.markdown("### Save Current Configuration")
        
        if st.session_state.get('engine_config'):
            filename = st.text_input(
                "Filename",
                value=f"{st.session_state.engine_config.engine_name.replace(' ', '_')}.yaml"
            )
            
            if st.button("💾 Save Configuration", type="primary"):
                # In a real implementation, this would save to file
                st.success(f"Configuration would be saved to: {filename}")
                st.info("File download functionality would be implemented here.")
        else:
            st.warning("No configuration to save.")
    
    with tab_load:
        st.markdown("### Load Configuration")
        
        uploaded = st.file_uploader(
            "Upload YAML Configuration",
            type=['yaml', 'yml'],
            help="Select a previously saved engine configuration file"
        )
        
        if uploaded:
            try:
                import yaml
                from rocket_engine.core.config import EngineConfig
                
                data = yaml.safe_load(uploaded)
                config = EngineConfig._from_nested_dict(data)
                
                st.success(f"Loaded: **{config.engine_name}**")
                
                if st.button("✅ Use This Configuration"):
                    st.session_state.engine_config = config
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab_export:
        st.markdown("### Export Options")
        
        if st.session_state.get('design_result'):
            st.markdown("#### Available Exports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reports**")
                st.button("📄 PDF Report", use_container_width=True)
                st.button("📊 Excel Summary", use_container_width=True)
                
            with col2:
                st.markdown("**Data Files**")
                st.button("📈 CSV Profile Data", use_container_width=True)
                st.button("📐 DXF Contour", use_container_width=True)
                st.button("🔧 YAML Config", use_container_width=True)
        else:
            st.info("Complete an analysis to enable exports.")
