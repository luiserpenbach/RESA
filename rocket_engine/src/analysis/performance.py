import numpy as np
import matplotlib.pyplot as plt

from rocket_engine.src.physics.combustion import CEASolver


def plot_cstar_contour(cea_solver: CEASolver,
                       nominal_pc_bar: float,
                       nominal_mr: float,
                       mr_range: tuple = (1.5, 8.0),
                       pc_range: tuple = (10.0, 100.0),
                       resolution: int = 100):
    """
    Plots a contour map of C* over Mixture Ratio and Chamber Pressure.
    """
    print("Generating C* Contour Map...")

    # Create grids
    mrs = np.linspace(mr_range[0], mr_range[1], resolution)
    pcs = np.linspace(pc_range[0], pc_range[1], resolution)
    MR, PC = np.meshgrid(mrs, pcs)

    CSTAR = np.zeros_like(MR)

    # Loop to calculate C*
    for i in range(resolution):
        for j in range(resolution):
            # eps doesn't affect C*, so just use 10
            res = cea_solver.run(pc_bar=PC[i, j], mr=MR[i, j], eps=10.0)
            CSTAR[i, j] = res.cstar

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour Fill
    cf = ax.contourf(MR, PC, CSTAR, levels=50, cmap='viridis')
    cbar = plt.colorbar(cf, ax=ax, label='Characteristic Velocity c* [m/s]')

    # Contour Lines
    cs = ax.contour(MR, PC, CSTAR, levels=15, colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8)

    # Nominal Point Markers
    ax.axvline(nominal_mr, color='white', linestyle='--', linewidth=1.5, label='Nominal MR')
    ax.axhline(nominal_pc_bar, color='white', linestyle=':', linewidth=1.5, label='Nominal Pc')
    ax.scatter(nominal_mr, nominal_pc_bar, color='red', s=100, marker='X', zorder=10, label='Operating Point')

    # Calculate Nominal C* for label
    nom_res = cea_solver.run(nominal_pc_bar, nominal_mr, eps=4.0)
    ax.text(nominal_mr + 0.1, nominal_pc_bar + 1, f"c* = {nom_res.cstar:.1f} m/s",
            color='white', fontweight='bold')

    ax.set_xlabel('Mixture Ratio (O/F)')
    ax.set_ylabel('Chamber Pressure [bar]')
    ax.set_title('Characteristic Velocity (c*) Performance Map (Eth90/N2O)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def plot_isp_vs_mr(cea_solver: CEASolver,
                   nominal_pc_bar: float,
                   nominal_mr: float,
                   pc_levels: list = [10, 20, 30, 50],
                   mr_range: tuple = (1.0, 8.0),
                   expansion_ratio: float = 40.0):
    """
    Plots Vacuum Isp vs Mixture Ratio for multiple chamber pressures.
    """
    print("Generating Isp Sensitivity Curves...")

    mrs = np.linspace(mr_range[0], mr_range[1], 50)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curves for each pressure level
    for pc in pc_levels:
        isps = []
        for mr in mrs:
            res = cea_solver.run(pc_bar=pc, mr=mr, eps=expansion_ratio)
            isps.append(res.isp_opt)

        style = '-' if pc == nominal_pc_bar else '--'
        width = 2.5 if pc == nominal_pc_bar else 1.5
        label = f'Pc = {pc} bar' + (' (Nominal)' if pc == nominal_pc_bar else '')

        ax.plot(mrs, isps, linestyle=style, linewidth=width, label=label)

    # Mark Nominal Point
    nom_res = cea_solver.run(nominal_pc_bar, nominal_mr, eps=expansion_ratio)
    ax.scatter(nominal_mr, nom_res.isp_opt, color='red', s=100, zorder=10)
    ax.annotate(f"Nominal\n{nom_res.isp_opt:.1f} s",
                (nominal_mr, nom_res.isp_opt),
                textcoords="offset points", xytext=(0, 15), ha='center', color='red')

    ax.axvline(nominal_mr, color='k', linestyle=':', alpha=0.5)

    ax.set_xlabel('Mixture Ratio (O/F)')
    ax.set_ylabel(f'SL Specific Impulse [s] (eps={expansion_ratio})')
    ax.set_title('Isp Sensitivity to Mixture Ratio & Chamber Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig