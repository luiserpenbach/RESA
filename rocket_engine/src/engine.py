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
from analysis.fluid_state import plot_n2o_t_rho_diagram, plot_n2o_p_t_diagram


@dataclass
class EngineConfig:
    """Master Configuration for the Liquid Rocket Engine."""
    # --- General ---
    engine_name: str
    fuel: str
    oxidizer: str
    thrust_n: float
    pc_bar: float
    mr: float

    # --- Nozzle / Chamber ---
    expansion_ratio: float = 0.0  # If 0, calc optimal for p_exit_bar
    p_exit_bar: float = 1.013

    L_star: float = 1100.0  # [mm]
    contraction_ratio: float = 10.0
    eff_combustion: float = 0.95
    theta_convergent: float = 30.0  # [deg]
    bell_fraction: float = 0.8

    # --- Cooling System ---
    coolant_name: str = "REFPROP::NitrousOxide"
    cooling_mode: str = 'counter-flow'

    # Dimensions (Throat)
    channel_width_throat: float = 1.0e-3
    channel_height: float = 0.75e-3
    rib_width_throat: float = 0.6e-3

    # Wall
    wall_thickness: float = 0.5e-3
    wall_roughness: float = 20e-6
    wall_conductivity: float = 15.0

    # Coolant Inlet
    coolant_p_in_bar: float = 98.0
    coolant_t_in_k: float = 290.0
    coolant_mass_fraction: float = 1.0


@dataclass
class EngineDesignResult:
    # Operating Point
    pc_bar: float
    mr: float

    # Performance
    isp_vac: float
    isp_sea: float
    thrust_vac: float
    thrust_sea: float
    massflow_total: float

    # Dimensions
    dt_mm: float
    de_mm: float
    length_mm: float
    expansion_ratio: float

    # Data Objects
    geometry: NozzleGeometryData
    channel_geometry: CoolingChannelGeometry
    cooling_data: dict

    # Physics Arrays
    mach_numbers: np.ndarray
    T_gas_recovery: np.ndarray
    h_gas: np.ndarray


class LiquidEngine:
    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.cea = CEASolver(config.fuel, config.oxidizer)
        self.last_result = None
        self.geo = None
        print(f"--- Initializing Engine: {config.engine_name} ---")

    def design(self, plot: bool = True) -> EngineDesignResult:
        """
        Sizing Mode: Calculates geometry to match Target Thrust at Target Pc.
        """
        print(f"\n>>> RUNNING DESIGN POINT (Thrust={self.cfg.thrust_n}N, Pc={self.cfg.pc_bar}bar)")

        # 1. Combustion & Optimal Expansion
        comb_init = self.cea.run(self.cfg.pc_bar, self.cfg.mr, eps=10.0)

        if self.cfg.expansion_ratio > 0.1:
            eps_design = self.cfg.expansion_ratio
        else:
            p_target = self.cfg.p_exit_bar if self.cfg.p_exit_bar > 0.001 else 1.013
            eps_design = get_expansion_ratio(p_target * 1e5, self.cfg.pc_bar * 1e5, comb_init.gamma)
            print(f"   Calculated Optimal Expansion Ratio: {eps_design:.2f}")

        # Rerun CEA with final Area Ratio
        comb_data = self.cea.run(self.cfg.pc_bar, self.cfg.mr, eps=eps_design)

        # 2. Throat Sizing (The definition of "Design")
        cstar_real = comb_data.cstar * self.cfg.eff_combustion
        isp_design = comb_data.isp_opt * self.cfg.eff_combustion  # Use optimal/target altitude Isp

        # mdot = F / (Isp * g0)
        mdot_total = self.cfg.thrust_n / (isp_design * Units.g0)

        # At = (mdot * cstar) / Pc
        At_m2 = (mdot_total * cstar_real) / (self.cfg.pc_bar * 1e5)
        Rt_m = np.sqrt(At_m2 / np.pi)

        print(f"   Design Mass Flow: {mdot_total:.3f} kg/s")
        print(f"   Throat Radius:    {Rt_m * 1000:.2f} mm")

        # 3. Generate Geometry
        nozzle_gen = NozzleGenerator(Rt_m * 1000, eps_design)
        self.geo = nozzle_gen.generate(
            contraction_ratio=self.cfg.contraction_ratio,
            L_star=self.cfg.L_star,
            bell_fraction=self.cfg.bell_fraction,
            theta_convergent=self.cfg.theta_convergent,
            R_ent_factor=1.0
        )

        # 4. Generate Channel Geometry
        xs = self.geo.x_full / 1000.0  # mm -> m
        ys = self.geo.y_full / 1000.0

        chan_gen = ChannelGeometryGenerator(
            x_contour=xs,
            y_contour=ys,
            wall_thickness=self.cfg.wall_thickness,
            roughness=self.cfg.wall_roughness
        )

        # Define channels at the throat
        self.channel_geo = chan_gen.define_by_throat_dimensions(
            width_at_throat=self.cfg.channel_width_throat,
            rib_at_throat=self.cfg.rib_width_throat,
            height=self.cfg.channel_height
        )

        # 5. Run Physics Loop
        return self._run_physics_analysis(
            pc_bar=self.cfg.pc_bar,
            mr=self.cfg.mr,
            mdot_total=mdot_total,
            comb_data=comb_data,
            plot=plot,
            label="Design"
        )

    def analyze(self, pc_bar: float, mr: float, plot: bool = True) -> EngineDesignResult:
        """
        Off-Design Mode: Simulates existing geometry at new Pc/MR.
        """
        if self.geo is None:
            raise RuntimeError("Run design() first to define engine geometry!")

        print(f"\n>>> RUNNING OFF-DESIGN (Pc={pc_bar}bar, MR={mr})")

        # 1. New Combustion State
        # Area ratio is fixed from design!
        # We need to recover it from the geometry
        Rt_mm = self.geo.y_throat_div[0]  # Throat radius
        Re_mm = self.geo.y_bell[-1]  # Exit radius
        eps_fixed = (Re_mm / Rt_mm) ** 2

        comb_data = self.cea.run(pc_bar, mr, eps=eps_fixed)

        # 2. Calculate Off-Design Mass Flow
        # Geometry is fixed (At constant), so mdot depends on new Pc and Cstar
        # mdot = (Pc * At) / cstar
        cstar_real = comb_data.cstar * self.cfg.eff_combustion

        At_m2 = np.pi * (Rt_mm / 1000.0) ** 2
        mdot_new = (pc_bar * 1e5 * At_m2) / cstar_real

        print(f"   Fixed Throat R:   {Rt_mm:.2f} mm")
        print(f"   New Mass Flow:    {mdot_new:.3f} kg/s")

        # 3. Run Physics Loop
        return self._run_physics_analysis(
            pc_bar=pc_bar,
            mr=mr,
            mdot_total=mdot_new,
            comb_data=comb_data,
            plot=plot,
            label=f"OffDesign_{pc_bar}bar"
        )

    def _run_physics_analysis(self, pc_bar, mr, mdot_total, comb_data, plot=False, label="Run"):
        """Shared physics logic for Design and Analysis modes."""

        # A. Discretization
        xs = self.geo.x_full / 1000.0
        ys = self.geo.y_full / 1000.0

        # B. Gas Dynamics (Mach Loop)
        # Recalculate Machs because Gamma changed with new MR
        areas = np.pi * ys ** 2
        at = np.min(areas)
        area_ratios = areas / at

        machs = []
        for i, ar in enumerate(area_ratios):
            if ar < 1.0001: ar = 1.0001
            supersonic = True if xs[i] > 0.001 else False
            try:
                m = mach_from_area_ratio(ar, comb_data.gamma, supersonic=supersonic)
            except:
                m = 0.0
            machs.append(m)
        machs = np.array(machs)

        # C. Heat Transfer Coefficients
        cstar_real = comb_data.cstar * self.cfg.eff_combustion

        h_g = calculate_bartz_coefficient(
            diameters=ys * 2,
            mach_numbers=machs,
            pc_pa=pc_bar * 1e5,
            c_star_mps=cstar_real,
            d_throat_m=np.sqrt(at / np.pi) * 2,
            T_combustion=comb_data.T_combustion,
            viscosity_gas=8.0e-5,  # TODO: Get from CEA if possible
            cp_gas=2200.0,
            prandtl_gas=0.68,
            gamma=comb_data.gamma
        )

        T_aw = calculate_adiabatic_wall_temp(comb_data.T_combustion, comb_data.gamma, machs)

        # D. Cooling Solver
        solver = RegenCoolingSolver(
            self.cfg.coolant_name,
            self.channel_geo,
            wall_conductivity=self.cfg.wall_conductivity
        )

        # Partition Mass Flow
        if "Nitrous" in self.cfg.coolant_name or "Oxygen" in self.cfg.coolant_name:
            mdot_coolant = mdot_total * (mr / (1 + mr))
        else:
            mdot_coolant = mdot_total * (1 / (1 + mr))

        mdot_coolant *= self.cfg.coolant_mass_fraction

        cooling_res = solver.solve(
            mdot_coolant_total=mdot_coolant,
            pin_coolant=self.cfg.coolant_p_in_bar * 1e5,
            tin_coolant=self.cfg.coolant_t_in_k,
            T_gas_recovery=T_aw,
            h_gas=h_g,
            mode=self.cfg.cooling_mode
        )

        # E. Package Results
        # Calculate Thrusts
        # F = mdot * Isp * g0
        f_vac = mdot_total * comb_data.isp_vac * Units.g0
        f_sea = mdot_total * comb_data.isp_opt * Units.g0  # Approx for sea level/optimum

        result = EngineDesignResult(
            pc_bar=pc_bar,
            mr=mr,
            isp_vac=comb_data.isp_vac,
            isp_sea=comb_data.isp_opt,
            thrust_vac=f_vac,
            thrust_sea=f_sea,
            massflow_total=mdot_total,

            dt_mm=np.sqrt(at / np.pi) * 2000,
            de_mm=ys[-1] * 2000,
            length_mm=xs[-1] * 1000 - xs[0] * 1000,
            expansion_ratio=(ys[-1] / np.sqrt(at / np.pi)) ** 2,

            geometry=self.geo,
            channel_geometry=self.channel_geo,
            cooling_data=cooling_res,
            mach_numbers=machs,
            T_gas_recovery=T_aw,
            h_gas=h_g
        )

        self.last_result = result  # Update state

        if plot:
            self._plot_results(xs, ys, cooling_res, title=f"Analysis: {label}")

        return result

    def save_specification(self, output_dir: str = "output", tag: str = None):
        """
        Saves the current engine state (scalars and profiles) to CSV files.

        Args:
            output_dir: Directory to save files.
            tag: Optional suffix for filenames (e.g. "design", "throttled_15bar").
                 If None, uses a timestamp.
        """
        if self.last_result is None:
            print("Error: No result to save. Run design() or analyze() first.")
            return

        res = self.last_result
        cfg = self.cfg
        cool = res.cooling_data
        geo = res.geometry
        chan = res.channel_geometry

        os.makedirs(output_dir, exist_ok=True)

        # Determine filename base
        if tag:
            base_name = f"{output_dir}/{cfg.engine_name}_{tag}"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{output_dir}/{cfg.engine_name}_{timestamp}"

        # --- 1. Scalar Specification Data ---
        # We group data into sections for readability
        spec_data = {
            "META": {
                "Engine Name": cfg.engine_name,
                "Run Tag": tag if tag else "timestamped",
                "Fuel": cfg.fuel,
                "Oxidizer": cfg.oxidizer,
                "Coolant": cfg.coolant_name
            },
            "OPERATING POINT": {
                "Chamber Pressure [bar]": round(res.pc_bar, 2),
                "Mixture Ratio": round(res.mr, 2),
                "Mass Flow Total [kg/s]": round(res.massflow_total, 4)
            },
            "PERFORMANCE": {
                "Thrust Vac [N]": round(res.thrust_vac, 1),
                "Thrust Sea [N]": round(res.thrust_sea, 1),
                "Isp Vac [s]": round(res.isp_vac, 1),
                "Isp Sea [s]": round(res.isp_sea, 1),
                "Expansion Ratio": round(res.expansion_ratio, 2),
                "Exit Diameter [mm]": round(res.de_mm, 2)
            },
            "GEOMETRY (Fixed)": {
                "Throat Diameter [mm]": round(res.dt_mm, 2),
                "Chamber Length [mm]": round(res.length_mm, 1),
                "Contraction Ratio": round(cfg.contraction_ratio, 1)
            },
            "COOLING SUMMARY": {
                "Max Hot Wall Temp [K]": round(np.max(cool['T_wall_hot']), 1),
                "Max Coolant Temp [K]": round(np.max(cool['T_coolant']), 1),
                "Max Heat Flux [MW/m2]": round(np.max(cool['q_flux']) / 1e6, 2),
                "Pressure Drop [bar]": round((np.max(cool['P_coolant']) - np.min(cool['P_coolant'])) / 1e5, 2),
                "Min Coolant Density [kg/m3]": round(np.min(cool['density']), 1)
            }
        }

        # Flatten dictionary for CSV writing
        flat_list = []
        for category, items in spec_data.items():
            flat_list.append({"Parameter": f"--- {category} ---", "Value": ""})
            for key, val in items.items():
                flat_list.append({"Parameter": key, "Value": val})

        df_spec = pd.DataFrame(flat_list)
        spec_file = f"{base_name}_spec.csv"
        df_spec.to_csv(spec_file, index=False)
        print(f"Saved Spec Sheet:   {spec_file}")

        # --- 2. Profile Data (1D Arrays) ---
        profile_data = {
            "Position X [mm]": geo.x_full,
            "Radius Y [mm]": geo.y_full,

            # Channel Geometry
            "Channel Width [mm]": chan.channel_width * 1000,
            "Channel Height [mm]": chan.channel_height * 1000,
            "Rib Width [mm]": chan.rib_width * 1000,
            "Helix Angle [deg]": np.degrees(chan.helix_angle),

            # Gas Physics
            "Mach Number": res.mach_numbers,
            "T_Gas_Recovery [K]": res.T_gas_recovery,
            "h_Gas [W/m2K]": res.h_gas,

            # Cooling Results
            "T_Wall_Hot [K]": cool['T_wall_hot'],
            "T_Wall_Cold [K]": cool['T_wall_cold'],
            "T_Coolant [K]": cool['T_coolant'],
            "P_Coolant [bar]": cool['P_coolant'] / 1e5,
            "Density_Coolant [kg/m3]": cool['density'],
            "Velocity_Coolant [m/s]": cool['velocity'],
            "Quality_Coolant": cool['quality'],
            "Heat_Flux [MW/m2]": cool['q_flux'] / 1e6
        }

        df_profile = pd.DataFrame(profile_data)
        profile_file = f"{base_name}_profile.csv"
        df_profile.to_csv(profile_file, index=False)
        print(f"Saved Profile Data: {profile_file}")

    def _plot_results(self, xs, ys, cooling_res, title="Engine Analysis"):
        # Same plotting logic as before, just updated title
        fig, axes = plt.subplots(4, 1, figsize=(10, 20))
        (ax1, ax3, ax5, ax7) = axes

        fig.suptitle(title, fontsize=16)

        # 1. Geometry & Temps
        ax1.plot(xs * 1000, ys * 1000, 'k-', linewidth=2, label="Wall")
        ax1.set_ylabel("Radius [mm]")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(xs * 1000, cooling_res['T_wall_hot'], 'r-', label="T_HotWall")
        ax2.plot(xs * 1000, cooling_res['T_coolant'], 'b--', label="T_Coolant")
        ax2.set_ylabel("Temperature [K]")
        ax2.legend(loc='upper right')

        # 2. Mach & Heat Flux
        ax3.plot(xs * 1000, self.last_result.mach_numbers, 'g-', label="Mach")
        ax3.set_ylabel("Mach Number")
        ax3.grid(True)

        ax4 = ax3.twinx()
        ax4.plot(xs * 1000, cooling_res['q_flux'] / 1e6, 'm-', label="Heat Flux")
        ax4.set_ylabel("Heat Flux [MW/m2]")

        # 3. Pressure
        ax5.plot(xs * 1000, cooling_res['P_coolant'] / 1e5, 'c-', label="Pressure")
        ax5.set_ylabel("Coolant Pressure [bar]")
        ax5.grid(True)

        # 4. Hydraulics
        ax7.plot(xs * 1000, cooling_res['velocity'], 'k-', label="Velocity")
        ax7.set_ylabel("Velocity [m/s]")
        ax7.set_xlabel("Axial Position [mm]")
        ax7.grid(True)

        ax8 = ax7.twinx()
        ax8.plot(xs * 1000, cooling_res['density'], 'b:', label="Density")
        ax8.set_ylabel("Density [kg/m3]")

        plt.tight_layout()
        plt.show()
        '''
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
        )'''


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define Design Point
    conf = EngineConfig(
        engine_name="HOPPER E1-1A",
        fuel="Ethanol90",
        oxidizer="N2O",
        thrust_n=2200,
        pc_bar=25,
        mr=4.0,
        p_exit_bar=1.013,
        contraction_ratio=12,
        expansion_ratio=8,

        # Geometry
        channel_width_throat=1.0e-3,
        channel_height=0.75e-3,
        rib_width_throat=0.6e-3,
        wall_thickness=0.5e-3,

        # Coolant
        coolant_p_in_bar=98.0
    )

    engine = LiquidEngine(conf)

    # 2. Design Point Run
    res_design = engine.design(plot=False)  # Don't popup individual plots

    # 3. Off-Design Runs
    # Case A: Throttled (Low Pc, Low Flow)
    res_throttle = engine.analyze(pc_bar=15.0, mr=3.5, plot=False)
    # Case B: Hot Run (High MR, slightly higher Pc)
    res_hot = engine.analyze(pc_bar=28.0, mr=5.5, plot=False)

    # 4. Compare in Phase Diagram
    print("\nGenerating Multi-Run Phase Diagram...")

    results_map = {
        "Design (25 bar, MR 4.0)": res_design,
        "Throttled (15 bar, MR 3.5)": res_throttle,
        "Hot Run (28 bar, MR 5.5)": res_hot
    }

    plot_n2o_p_t_diagram(results_map)




