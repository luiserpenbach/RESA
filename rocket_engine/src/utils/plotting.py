from matplotlib import pyplot as plt


def plot_chamber_with_mach(x, y, mach, title="Thrust Chamber", throat_x=None):
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(x, y, color='peachpuff', lw=2, label='Inner Wall')
    ax1.plot(x, y, color='peachpuff', lw=2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Axial Position [mm]')
    ax1.set_ylabel('Radius [mm]')
    if throat_x is not None:
        ax1.axvline(throat_x, color='gray', linestyle='--', alpha=0.7, label='Throat')

    ax2 = ax1.twinx()
    ax2.plot(x, mach, color='cyan', lw=2, label='Mach Number')
    ax2.set_ylabel('Mach Number', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()