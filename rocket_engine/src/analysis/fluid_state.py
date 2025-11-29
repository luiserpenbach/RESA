import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from matplotlib.colors import LogNorm


def plot_n2o_t_rho_diagram(engine_result, fluid_name="NitrousOxide"):
    """
    Plots a T-rho diagram with a full Pressure Gradient background.
    """

    # 1. Setup Phase Dome
    try:
        T_trip = CP.PropsSI('T_triple', fluid_name)
        T_crit = CP.PropsSI('T_critical', fluid_name)
        Rho_crit = CP.PropsSI('rhocrit', fluid_name)
        P_crit = CP.PropsSI('pcrit', fluid_name)
    except:
        # Fallback for fluids where triple point isn't defined in standard interface
        T_trip = 182.3  # N2O
        T_crit = 309.52
        Rho_crit = 452.0
        P_crit = 72.45e5

    # Saturation Line
    T_dome = np.linspace(T_trip, T_crit - 0.1, 200)
    rho_liq = np.array([CP.PropsSI('D', 'T', t, 'Q', 0, fluid_name) for t in T_dome])
    rho_vap = np.array([CP.PropsSI('D', 'T', t, 'Q', 1, fluid_name) for t in T_dome])

    # 2. Generate Background Pressure Grid
    # We span the plot range
    T_min, T_max = 200, 450
    rho_min, rho_max = 0, 1500

    # Create mesh
    resolution = 400
    rho_grid = np.linspace(rho_min, rho_max, resolution)
    t_grid = np.linspace(T_min, T_max, resolution)
    RHO, T = np.meshgrid(rho_grid, t_grid)

    # Calculate Pressure at every point (Vectorized usually works, else loop)
    # Using HEOS backend, PropsSI isn't natively vectorized for 2D arrays, need flat loop
    # Optimization: Use 'P' from T,D

    # Flatten for calculation
    rho_flat = RHO.flatten()
    t_flat = T.flatten()

    # CP.PropsSI is slow in loops.
    # vectorization hack:
    # P_flat = CP.PropsSI('P', 'T', t_flat, 'D', rho_flat, fluid_name)
    # (If this fails, we assume loop)
    try:
        P_flat = CP.PropsSI('P', 'T', t_flat, 'D', rho_flat, fluid_name)
        P_grid = P_flat.reshape(RHO.shape)
    except:
        # Fallback to loop if vectorization fails on your system
        P_grid = np.zeros_like(RHO)
        for i in range(resolution):
            for j in range(resolution):
                try:
                    P_grid[i, j] = CP.PropsSI('P', 'T', t_grid[i], 'D', rho_grid[j], fluid_name)
                except:
                    P_grid[i, j] = np.nan

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 9))

    # A. Background Pressure Contour
    # Use LogNorm because pressure varies from 1 bar to >100 bar
    levels = np.logspace(np.log10(1e5), np.log10(300e5), 50)

    cf = ax.contourf(RHO, T, P_grid, levels=levels, cmap='viridis', norm=LogNorm(), alpha=0.8)

    # Add contour lines for clarity
    cl = ax.contour(RHO, T, P_grid, levels=[10e5, 20e5, 50e5, 72.4e5, 100e5, 200e5],
                    colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cl, fmt=lambda x: f'{x / 1e5:.0f} bar', inline=True, fontsize=8)

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, label='Fluid Pressure [Pa]')
    # Fix ticks to be readable bars
    cbar.set_ticks([1e5, 10e5, 50e5, 100e5, 200e5])
    cbar.set_ticklabels(['1 bar', '10 bar', '50 bar', '100 bar', '200 bar'])

    # B. Phase Dome
    ax.plot(rho_liq, T_dome, 'w-', linewidth=2)  # White lines stand out against dark background
    ax.plot(rho_vap, T_dome, 'w-', linewidth=2)
    ax.plot(Rho_crit, T_crit, 'wo', markersize=6, label='Critical Point')
    ax.text(Rho_crit + 20, T_crit, 'Critical Point', color='white', fontsize=9, fontweight='bold')

    # C. Coolant Path
    cool = engine_result.cooling_data
    path_T = cool['T_coolant']
    path_rho = cool['density']

    # Plot Path as points with white edges to be visible
    sc = ax.scatter(path_rho, path_T, c='white', s=40, edgecolors='k', zorder=10, label='Cooling Path')

    # Connect dots
    ax.plot(path_rho, path_T, 'w--', linewidth=1, alpha=0.7)

    # Start/End Labels
    ax.annotate('Inlet', xy=(path_rho[0], path_T[0]), xytext=(path_rho[0], path_T[0] - 15),
                color='white', ha='center',
                arrowprops=dict(facecolor='white', arrowstyle='->'))

    ax.annotate('Outlet', xy=(path_rho[-1], path_T[-1]), xytext=(path_rho[-1], path_T[-1] + 15),
                color='white', ha='center',
                arrowprops=dict(facecolor='white', arrowstyle='->'))

    # D. Styling
    ax.set_xlabel(r'Density $\rho$ [kg/m$^3$]')
    ax.set_ylabel(r'Temperature $T$ [K]')
    ax.set_title(f'Nitrous Oxide State Diagram (Background = Pressure)')

    # Limits
    ax.set_xlim(0, 1250)
    ax.set_ylim(200, 360)

    # Add text for regions
    ax.text(200, 340, "Supercritical Fluid", color='white', alpha=0.8, fontsize=12, fontweight='bold')
    ax.text(1000, 220, "Liquid", color='white', alpha=0.8, fontsize=12, fontweight='bold')
    ax.text(100, 220, "Vapor", color='white', alpha=0.8, fontsize=12, fontweight='bold')
    ax.text(600, 280, "Two-Phase", color='white', alpha=0.8, ha='center')

    plt.tight_layout()
    plt.show()


def plot_n2o_p_t_diagram(engine_result, fluid_name="NitrousOxide"):
    """
    Plots a Pressure-Temperature (P-T) diagram with Density background.
    """

    # 1. Setup Phase Boundaries
    # -------------------------
    try:
        T_trip = CP.PropsSI('T_triple', fluid_name)
        T_crit = CP.PropsSI('T_critical', fluid_name)
        P_crit = CP.PropsSI('pcrit', fluid_name)
    except:
        T_trip = 182.33
        T_crit = 309.52
        P_crit = 72.45e5

    # Saturation Line (Vapor Pressure Curve)
    # Valid from Triple Point to Critical Point
    T_sat = np.linspace(T_trip, T_crit - 0.1, 200)
    P_sat = np.array([CP.PropsSI('P', 'T', t, 'Q', 0, fluid_name) for t in T_sat])

    # 2. Generate Background Density Grid
    # -----------------------------------
    # Span the plot range
    T_min, T_max = 200, 500
    P_min, P_max = 1e5, 120e5  # 1 bar to 120 bar

    resolution = 300
    t_grid = np.linspace(T_min, T_max, resolution)
    p_grid = np.linspace(P_min, P_max, resolution)
    T_MESH, P_MESH = np.meshgrid(t_grid, p_grid)

    # Calculate Density at every point (Rho = f(P,T))
    # This is often faster than P=f(T,Rho)
    # Using a loop since vectorization depends on CP version/backend
    Rho_grid = np.zeros_like(T_MESH)

    for i in range(resolution):
        for j in range(resolution):
            try:
                # If below saturation curve, it's gas/liquid depending on T
                # PropsSI handles single phase automatically given P,T
                Rho_grid[i, j] = CP.PropsSI('D', 'T', t_grid[j], 'P', p_grid[i], fluid_name)
            except:
                Rho_grid[i, j] = np.nan

    # 3. Plotting
    # -----------
    fig, ax = plt.subplots(figsize=(12, 9))

    # A. Background Density Contour
    # Density varies from ~1 kg/m3 (gas) to ~1200 kg/m3 (liquid)
    # Linear scale usually works fine for density color map
    levels = np.linspace(0, 1200, 50)

    cf = ax.contourf(T_MESH, P_MESH / 1e5, Rho_grid, levels=levels, cmap='Spectral', alpha=0.8)

    # Add contour lines for specific densities (Isochores)
    cl = ax.contour(T_MESH, P_MESH / 1e5, Rho_grid, levels=[100, 300, 500, 700, 900, 1100],
                    colors='k', linewidths=0.5, alpha=0.3)
    ax.clabel(cl, fmt=lambda x: f'{x:.0f} kg/m3', inline=True, fontsize=8)

    cbar = plt.colorbar(cf, ax=ax, label='Fluid Density [kg/m$^3$]')

    # B. Phase Lines
    # Saturation Curve
    ax.plot(T_sat, P_sat / 1e5, 'k-', linewidth=2.5, label='Saturation Line')

    # Critical Point
    ax.plot(T_crit, P_crit / 1e5, 'ko', markersize=8, label='Critical Point')

    # Critical Isobar and Isotherm (dashed)
    ax.axhline(P_crit / 1e5, color='k', linestyle=':', alpha=0.5)
    ax.axvline(T_crit, color='k', linestyle=':', alpha=0.5)

    # C. Coolant Path
    cool = engine_result.cooling_data
    path_T = cool['T_coolant']
    path_P = cool['P_coolant'] / 1e5  # bar

    # Plot Path
    sc = ax.scatter(path_T, path_P, c='white', s=10, edgecolors='k', zorder=10, label='Cooling Path')
    ax.plot(path_T, path_P, 'w--', linewidth=1.5)

    # Direction Arrows
    ax.annotate('Inlet', xy=(path_T[0], path_P[0]), xytext=(path_T[0] - 5, path_P[0] + 5),
                color='black', fontweight='bold', ha='center',
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    ax.annotate('Outlet', xy=(path_T[-1], path_P[-1]), xytext=(path_T[-1] + 5, path_P[-1] - 5),
                color='black', fontweight='bold', ha='center',
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # D. Styling
    ax.set_xlabel(r'Temperature $T$ [K]')
    ax.set_ylabel(r'Pressure $P$ [bar]')
    ax.set_title(f'{fluid_name} P-T Phase Diagram (Background = Density)')

    # Region Labels
    ax.text(220, 100, "LIQUID", fontsize=14, fontweight='bold', alpha=0.3, color='blue')
    ax.text(320, 20, "GAS", fontsize=14, fontweight='bold', alpha=0.3, color='red')
    ax.text(320, 90, "SUPERCRITICAL", fontsize=14, fontweight='bold', alpha=0.3, color='purple')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, color='k', linestyle='--')

    # Ensure limits cover the path
    ax.set_ylim(0, max(110, np.max(path_P) * 1.1))
    ax.set_xlim(200, max(350, np.max(path_T) * 1.05))

    plt.tight_layout()
    plt.show()