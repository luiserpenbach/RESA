import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rocket_engine.src.physics.combustion import CEASolver


def plot_cstar_contour(cea_solver: CEASolver,
                       nominal_pc_bar: float,
                       nominal_mr: float,
                       mr_range: tuple = (1.5, 8.0),
                       pc_range: tuple = (10.0, 100.0),
                       resolution: int = 50,
                       trajectory: list = None,
                        envelope: dict = None,
                       show: bool = True):
    """
    Plots a contour map of C* over Mixture Ratio and Chamber Pressure, able to overlay a throttle trajectory

    Args:
        envelope: Dict with limits {'pc_min': 15, 'pc_max': 30, 'mr_min': 3.0, 'mr_max': 5.0}
    """
    print("Generating C* Contour Map...")

    # 1. Create data grid
    mrs = np.linspace(mr_range[0], mr_range[1], resolution)
    pcs = np.linspace(pc_range[0], pc_range[1], resolution)
    MR, PC = np.meshgrid(mrs, pcs)

    CSTAR = np.zeros_like(MR)

    # Loop to fill grid with C* values
    for i in range(resolution):
        for j in range(resolution):
            # eps doesn't affect C*, so just use 10
            res = cea_solver.run(pc_bar=PC[i, j], mr=MR[i, j], eps=10.0)
            CSTAR[i, j] = res.cstar

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour Fill
    cf = ax.contourf(MR, PC, CSTAR, levels=50, cmap='viridis')
    cbar = plt.colorbar(cf, ax=ax, label='Characteristic Velocity c* [m/s]')
    # Contour Lines
    cs = ax.contour(MR, PC, CSTAR, levels=15, colors='k', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8)

    # 3. Draw Operating Envelope Box if given
    if envelope:
        pc_min = envelope.get('pc_min', pc_range[0])
        pc_max = envelope.get('pc_max', pc_range[1])
        mr_min = envelope.get('mr_min', mr_range[0])
        mr_max = envelope.get('mr_max', mr_range[1])

        width = mr_max - mr_min
        height = pc_max - pc_min

        # Draw the shaded box
        rect = Rectangle((mr_min, pc_min), width, height,
                         linewidth=2, edgecolor='grey', facecolor='grey', alpha=0.25,
                         label='Target Operating Envelope', linestyle='--')
        ax.add_patch(rect)

        # Add corner labels
        ax.text(mr_min, pc_max + 1, "Limit", color='cyan', fontsize=9, fontweight='bold')

    # 4. Plot Nominal Point Markers
    ax.axvline(nominal_mr, color='white', linestyle='--', linewidth=1.5, label='Nominal MR')
    ax.axhline(nominal_pc_bar, color='white', linestyle=':', linewidth=1.5, label='Nominal Pc')
    ax.scatter(nominal_mr, nominal_pc_bar, color='red', s=100, marker='X', zorder=10, label='Operating Point')

    # Calculate Nominal C* for label
    nom_res = cea_solver.run(nominal_pc_bar, nominal_mr, eps=10.0)
    ax.text(nominal_mr + 0.1, nominal_pc_bar + 1, f"c* = {nom_res.cstar:.1f} m/s",
            color='white', fontweight='bold')

    # 5. Plot Trajectory if given
    if trajectory:
        t_mr = [p['mr'] for p in trajectory]
        t_pc = [p['pc'] for p in trajectory]
        # Draw path
        ax.plot(t_mr, t_pc, 'w-', linewidth=2.5, alpha=0.8, label='Throttle Path')
        ax.scatter(t_mr, t_pc, color='black', s=30, alpha=0.8)
        # Annotate Thrust at each point
        # We skip every other point if there are too many (>15) to avoid clutter
        step = 2 if len(trajectory) > 15 else 1
        # Arrows to show direction (High Pc -> Low Pc)
        # Assuming trajectory is ordered (e.g. 100% -> 40%)
        for i in range(0, len(trajectory), step):
            p = trajectory[i]
            mr_pt = p['mr']
            pc_pt = p['pc']
            thrust = p.get('thrust', 0.0)
            throttle_pct = p.get('pct', 0.0)

            # Format text: "1500 N (70%)"
            label_txt = f"{thrust:.0f} N"

            # Alternate label position (left/right) to reduce overlapping
            offset_x = 10 if i % 2 == 0 else -10
            ha = 'left' if i % 2 == 0 else 'right'
            ax.annotate(label_txt,
                        xy=(mr_pt, pc_pt),
                        xytext=(offset_x, 5),
                        textcoords="offset points",
                        color='white',
                        fontsize=9,
                        fontweight='bold',
                        ha=ha,
                        arrowprops=dict(arrowstyle="-", color='white', alpha=0.5))

    # Styling
    ax.set_xlabel('Mixture Ratio (O/F)')
    ax.set_ylabel('Chamber Pressure [bar]')
    ax.set_title('c* Performance Map (Eth90/N2O)')
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
    ax.set_ylabel(f'Sea-Level Specific Impulse [s] (eps={expansion_ratio:.3f})')
    ax.set_title('Isp Sensitivity to Mixture Ratio & Chamber Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # SAVE FIG??
    plt.show()
    return fig