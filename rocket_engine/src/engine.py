import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import Modules
from propellants import Propellant
from physics.combustion import CEASolver
from geometry.nozzle import NozzleGenerator, NozzleGeometryData
from geometry.cooling import ChannelGeometryGenerator, CoolingChannelGeometry
from physics.fluid_flow import mach_from_area_ratio
from physics.heat_transfer import calculate_bartz_coefficient, calculate_adiabatic_wall_temp
from physics.cooling import RegenCoolingSolver
from geometry.injector import SwirlInjectorSizer
from rocket_engine.src.visualization_3d import plot_engine_3d, plot_coolant_channels_3d
from utils.units import Units
from visualization import plot_channel_cross_section, plot_channel_cross_section_radial


@dataclass
class EngineConfig:
    engine_name: str
    fuel: str
    oxidizer: str
    thrust_n: float         # Design Thrust [N]
    pc_bar: float           # Chamber Pressure [Bar]
    mr: float               # Mixture Ratio (Ox/Fuel)
    p_exit_bar: float       # Target exit pressure
    L_star: float           # Characteristic Length [mm]
    contraction_ratio: float
    eff_combustion: float = 0.95


@dataclass
class EngineDesignResult:
    # High level metrics
    isp_vac: float
    thrust_vac: float
    massflow_total: float
    dt_mm: float
    de_mm: float
    length_mm: float

    # Detailed Data Objects
    geometry: NozzleGeometryData
    channel_geometry: CoolingChannelGeometry
    cooling_data: dict

    # --- ADDED PHYSICS ARRAYS ---
    mach_numbers: np.ndarray
    T_gas_recovery: np.ndarray
    h_gas: np.ndarray


class LiquidEngine:
    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.cea = CEASolver(config.fuel, config.oxidizer)

        print(f"--- Initializing Engine: {config.engine_name} ---")

    def design(self, plot: bool = True):
        # 1. Combustion Analysis
        p_amb = 1.013

        # Run CEA (Initial estimate)
        comb_data = self.cea.run(self.cfg.pc_bar, self.cfg.mr, eps=10.0)

        # Calculate Optimal Expansion
        from physics.fluid_flow import get_expansion_ratio
        eps_opt = get_expansion_ratio(p_amb * 1e5, self.cfg.pc_bar * 1e5, comb_data.gamma)
        print(f"Calculated Optimal Expansion Ratio: {eps_opt:.2f}")

        # Rerun CEA with correct Eps
        self.comb_data = self.cea.run(self.cfg.pc_bar, self.cfg.mr, eps=eps_opt)

        # 2. Throat Sizing
        cstar_real = self.comb_data.cstar * self.cfg.eff_combustion
        isp_real = self.comb_data.isp_opt * self.cfg.eff_combustion
        mdot_total = self.cfg.thrust_n / (isp_real * Units.g0)

        At_m2 = (mdot_total * cstar_real) / (self.cfg.pc_bar * 1e5)
        Rt_m = np.sqrt(At_m2 / np.pi)

        print(f"Total Mass Flow: {mdot_total:.3f} kg/s")
        print(f"Throat Radius: {Rt_m * 1000:.2f} mm")

        # 3. Generate Geometry
        nozzle_gen = NozzleGenerator(Rt_m * 1000, eps_opt)
        self.geo = nozzle_gen.generate(
            contraction_ratio=self.cfg.contraction_ratio,
            L_star=self.cfg.L_star,
            bell_fraction=0.8,
            theta_convergent=30,
            R_ent_factor=1.0
        )

        # 4. Cooling Analysis
        # 4a. Prepare 1D Discretization
        xs = self.geo.x_full / 1000.0  # mm -> m
        ys = self.geo.y_full / 1000.0  # mm -> m

        # Calculate local Area Ratios and Mach numbers
        areas = np.pi * ys ** 2
        at = np.pi * (Rt_m) ** 2
        area_ratios = areas / at

        machs = []
        for i, ar in enumerate(area_ratios):
            # Strict throat check to avoid floating point issues near 1.0
            if ar < 1.0001:
                ar = 1.0001

            # Throat is at x=0.
            supersonic = True if xs[i] > 0.001 else False

            try:
                m = mach_from_area_ratio(ar, self.comb_data.gamma, supersonic=supersonic)
            except Exception as e:
                print(f"Mach Calc Error at x={xs[i]:.4f}, AR={ar:.4f}: {e}")
                m = 0.0
            machs.append(m)
        machs = np.array(machs)

        # 4b. Calculate Gas Side Heat Transfer (Bartz)
        h_g = calculate_bartz_coefficient(
            diameters=ys * 2,
            mach_numbers=machs,
            pc_pa=self.cfg.pc_bar * 1e5,
            c_star_mps=cstar_real,
            d_throat_m=Rt_m * 2,
            T_combustion=self.comb_data.T_combustion,
            viscosity_gas=8.0e-5,  # Approx
            cp_gas=2200.0,  # Approx
            prandtl_gas=0.68,
            gamma=self.comb_data.gamma
        )

        T_aw = calculate_adiabatic_wall_temp(
            self.comb_data.T_combustion, self.comb_data.gamma, machs
        )

        # 4c. Setup Cooling Geometry
        chan_gen = ChannelGeometryGenerator(x_contour=xs, y_contour=ys, wall_thickness=0.5e-3)

        # Auto-size for 1.5mm width / 1mm rib at throat
        channel_geo = chan_gen.define_by_throat_dimensions(
            width_at_throat=1.0e-3,
            rib_at_throat=0.6e-3,
            height=0.75e-3
        )

        # 4d. Run Cooling Solver
        solver = RegenCoolingSolver("REFPROP::NitrousOxide", channel_geo, 15)

        mdot_ox = mdot_total * (self.cfg.mr / (1 + self.cfg.mr))

        cooling_res = solver.solve(
            mdot_coolant_total=mdot_ox,
            pin_coolant=100e5,
            tin_coolant=290.0,
            T_gas_recovery=T_aw,
            h_gas=h_g,
            mode='counter-flow'
        )

        # Store Result
        self.last_result = EngineDesignResult(
            isp_vac=self.comb_data.isp_vac,
            thrust_vac=self.cfg.thrust_n,
            massflow_total=mdot_total,
            dt_mm=Rt_m * 2000,
            de_mm=Rt_m * 2000 * np.sqrt(eps_opt),
            length_mm=xs[-1] * 1000 - xs[0] * 1000,
            geometry=self.geo,
            channel_geometry=channel_geo,
            cooling_data=cooling_res,
            # --- Store Physics Arrays ---
            mach_numbers=machs,
            T_gas_recovery=T_aw,
            h_gas=h_g
        )

        if plot:
            self._plot_results(xs, ys, cooling_res)

        return self.last_result

    def save_specification(self, output_dir: str = "output"):
        if not hasattr(self, 'last_result'):
            print("Error: Run design() first.")
            return

        res = self.last_result
        cfg = self.cfg
        cool = res.cooling_data
        geo = res.geometry
        chan = res.channel_geometry

        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{output_dir}/{cfg.engine_name.replace(' ', '_')}"

        # 1. Profile Data
        # Ensure we write the actual physics arrays now!
        profile_data = {
            "Position X [mm]": geo.x_full,
            "Radius Y [mm]": geo.y_full,
            "Mach Number": res.mach_numbers,  # <--- FIXED
            "T_Gas_Recovery [K]": res.T_gas_recovery,  # <--- FIXED
            "h_Gas [W/m2K]": res.h_gas,  # <--- ADDED
            "Channel Width [mm]": chan.channel_width * 1000,
            "Channel Height [mm]": chan.channel_height * 1000,
            "T_Wall_Hot [K]": cool['T_wall_hot'],
            "T_Wall_Cold [K]": cool['T_wall_cold'],
            "T_Coolant [K]": cool['T_coolant'],
            "P_Coolant [bar]": cool['P_coolant'] / 1e5,
            "Heat Flux [MW/m2]": cool['q_flux'] / 1e6,
            "Coolant Velocity [m/s]": cool['velocity'],
            "Coolant Quality": cool['quality']
        }

        df_profile = pd.DataFrame(profile_data)
        df_profile.to_csv(f"{base_name}_profile.csv", index=False)
        print(f"Saved Profile Data to {base_name}_profile.csv")

    def _plot_results(self, xs, ys, cooling_res):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10))

        # Top: Geometry & Temps
        ax1.plot(xs * 1000, ys * 1000, 'k-', linewidth=2, label="Wall")
        ax1.set_ylabel("Radius [mm]")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs * 1000, cooling_res['T_wall_hot'], 'r-', label="T_HotWall")
        ax2.plot(xs * 1000, cooling_res['T_coolant'], 'b--', label="T_Coolant")
        ax2.set_ylabel("Temperature [K]")
        ax2.legend(loc='upper right')

        # Bottom: Mach & Heat Flux
        ax3.plot(xs * 1000, self.last_result.mach_numbers, 'g-', label="Mach")
        ax3.set_ylabel("Mach Number")
        ax3.set_xlabel("Axial Position [mm]")
        ax3.grid(True)

        ax4 = ax3.twinx()
        ax4.plot(xs * 1000, cooling_res['q_flux'] / 1e6, 'm-', label="Heat Flux [MW/m2]")
        ax4.set_ylabel("Heat Flux [MW/m2]")

        plt.tight_layout()
        plt.show()

        # --- Visualize Throat Cross Section ---
        # Find index of throat (x closest to 0)
        idx_throat = np.abs(xs).argmin()

        # Plot Radial View (Full 360 degrees)
        plot_channel_cross_section_radial(
            self.last_result.channel_geometry,
            station_idx=idx_throat,
            closeout_thickness=0.001,
            sector_angle=360  # Change to 90 or 45 to zoom in
        )
        # --- 3D Visualization ---
        print("\nGenerating 3D Cutaway...")
        plot_engine_3d(
            self.last_result.channel_geometry,
            closeout_thickness=0.001,  # 1mm jacket
            sector_angle=270,  # 90 degree cutaway
            resolution=100  # Smoothness
        )

        print("\nGenerating 3D Channel Visualization...")
        plot_coolant_channels_3d(
            self.last_result.channel_geometry,
            num_channels_to_show=30,  # Plot 5 adjacent channels
            resolution=50
        )

# ... (Main block remains same) ...

# --- Execution Block (Example) ---
if __name__ == "__main__":
    # Define constraints
    conf = EngineConfig(
        engine_name="HOPPER E1-1A",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200,
        pc_bar=25,
        mr=4.0,
        p_exit_bar=1.013,
        L_star=1200,
        contraction_ratio=10.0
    )

    # Instantiate and Run
    engine = LiquidEngine(conf)
    result = engine.design()
    engine.save_specification()



    # Sizing Injector
    inj_sizer = SwirlInjectorSizer(result.massflow_total, p_drop=5e5, n_elements=5, propellant_density=800)
    inj_geo = inj_sizer.calculate()
    print(f"Injector Sizing: Orifice={inj_geo.orifice_radius:.2f}mm")