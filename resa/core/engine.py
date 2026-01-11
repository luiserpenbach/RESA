"""
Main Engine class for RESA.

This is the primary entry point for rocket engine design and analysis.
It orchestrates all solvers, geometry generators, and produces results.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

from resa.core.config import EngineConfig
from resa.core.results import (
    EngineDesignResult,
    CombustionResult,
    NozzleGeometry,
    CoolingChannelGeometry,
    CoolingResult,
    ThrottlePoint,
    ThrottleCurve,
)
from resa.core.exceptions import (
    ConfigurationError,
    CombustionError,
    CoolingError,
    GeometryError,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EngineComponents:
    """
    Container for engine solver components.

    This enables dependency injection for testing and
    swapping implementations.
    """
    combustion_solver: Any = None
    cooling_solver: Any = None
    nozzle_generator: Any = None
    channel_generator: Any = None
    fluid_provider: Any = None


class Engine:
    """
    Main rocket engine design and analysis class.

    This class orchestrates:
    - Combustion analysis (via CEA or other solvers)
    - Nozzle geometry generation
    - Cooling channel geometry
    - Regenerative cooling analysis
    - Performance calculations

    Example:
        >>> from resa import Engine, EngineConfig
        >>>
        >>> config = EngineConfig(
        ...     engine_name="Phoenix-1",
        ...     fuel="Ethanol90",
        ...     oxidizer="N2O",
        ...     thrust_n=2200,
        ...     pc_bar=25,
        ...     mr=4.0
        ... )
        >>>
        >>> engine = Engine(config)
        >>> result = engine.design()
        >>> print(f"Isp: {result.isp_vac:.1f} s")
    """

    def __init__(
        self,
        config: EngineConfig,
        components: Optional[EngineComponents] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize engine with configuration.

        Args:
            config: Engine configuration parameters
            components: Optional custom solver components
            output_dir: Optional output directory override
        """
        self.config = config
        self._output_dir = output_dir or f"./output/{config.engine_name}"

        # Validate configuration
        validation = config.validate()
        if not validation.is_valid:
            raise ConfigurationError(
                f"Invalid configuration: {', '.join(validation.errors)}"
            )
        for warning in validation.warnings:
            logger.warning(warning)

        # Initialize components (lazy loading)
        self._components = components
        self._combustion_solver = None
        self._cooling_solver = None
        self._nozzle_generator = None
        self._channel_generator = None
        self._fluid_provider = None

        # Cached geometry
        self._nozzle_geometry: Optional[NozzleGeometry] = None
        self._channel_geometry: Optional[CoolingChannelGeometry] = None

        # Last result
        self._last_result: Optional[EngineDesignResult] = None

        logger.info(f"Initialized Engine: {config.engine_name}")

    def _init_solvers(self):
        """Lazy initialization of solvers."""
        if self._combustion_solver is None:
            if self._components and self._components.combustion_solver:
                self._combustion_solver = self._components.combustion_solver
            else:
                # Import default solver
                from resa.solvers.combustion import CEASolver
                self._combustion_solver = CEASolver(
                    self.config.fuel,
                    self.config.oxidizer
                )

        if self._nozzle_generator is None:
            if self._components and self._components.nozzle_generator:
                self._nozzle_generator = self._components.nozzle_generator
            else:
                from resa.geometry.nozzle import NozzleGenerator
                self._nozzle_generator = NozzleGenerator()

        if self._channel_generator is None:
            if self._components and self._components.channel_generator:
                self._channel_generator = self._components.channel_generator
            else:
                from resa.geometry.cooling_channels import ChannelGeometryGenerator
                self._channel_generator = ChannelGeometryGenerator()

        if self._cooling_solver is None:
            if self._components and self._components.cooling_solver:
                self._cooling_solver = self._components.cooling_solver
            else:
                from resa.solvers.cooling import RegenCoolingSolver
                self._cooling_solver = RegenCoolingSolver()

        if self._fluid_provider is None:
            if self._components and self._components.fluid_provider:
                self._fluid_provider = self._components.fluid_provider
            else:
                from resa.physics.fluids import CoolPropFluid
                self._fluid_provider = CoolPropFluid(self.config.coolant_name)

    def design(self) -> EngineDesignResult:
        """
        Run full design point analysis.

        This calculates:
        1. Combustion properties from CEA
        2. Optimal/specified expansion ratio
        3. Throat sizing from thrust requirement
        4. Nozzle contour geometry
        5. Cooling channel geometry
        6. Gas dynamics along nozzle
        7. Regenerative cooling analysis

        Returns:
            Complete EngineDesignResult with all analysis data
        """
        self._init_solvers()

        logger.info(
            f"Running design: Thrust={self.config.thrust_n}N, "
            f"Pc={self.config.pc_bar}bar, MR={self.config.mr}"
        )

        # Step 1: Combustion analysis
        combustion = self._run_combustion()

        # Step 2: Calculate expansion ratio
        if self.config.expansion_ratio > 0.1:
            eps_design = self.config.expansion_ratio
        else:
            eps_design = self._calculate_optimal_expansion(combustion.gamma)
            logger.info(f"Calculated optimal expansion ratio: {eps_design:.2f}")

        # Re-run combustion with final expansion ratio
        combustion = self._combustion_solver.run(
            self.config.pc_bar,
            self.config.mr,
            eps_design
        )

        # Step 3: Size throat
        throat_radius, mass_flow = self._size_throat(combustion)

        # Step 4: Generate nozzle geometry
        self._nozzle_geometry = self._generate_nozzle(throat_radius, eps_design)

        # Step 5: Generate channel geometry
        self._channel_geometry = self._generate_channels()

        # Step 6: Gas dynamics analysis
        mach_numbers, T_gas_recovery, h_gas = self._analyze_gas_dynamics(combustion)

        # Step 7: Cooling analysis
        cooling = self._run_cooling(mass_flow, T_gas_recovery, h_gas)

        # Build result
        result = EngineDesignResult(
            timestamp=datetime.now(),
            run_type="design",
            pc_bar=self.config.pc_bar,
            mr=self.config.mr,
            isp_vac=combustion.isp_vac * self.config.eff_combustion,
            isp_sea=combustion.isp_opt * self.config.eff_combustion,
            thrust_vac=mass_flow * combustion.isp_vac * self.config.eff_combustion * 9.80665,
            thrust_sea=mass_flow * combustion.isp_opt * self.config.eff_combustion * 9.80665,
            massflow_total=mass_flow,
            massflow_ox=mass_flow * self.config.mr / (1 + self.config.mr),
            massflow_fuel=mass_flow / (1 + self.config.mr),
            dt_mm=throat_radius * 2000,
            de_mm=throat_radius * np.sqrt(eps_design) * 2000,
            length_mm=self._nozzle_geometry.total_length,
            expansion_ratio=eps_design,
            combustion=combustion,
            nozzle_geometry=self._nozzle_geometry,
            channel_geometry=self._channel_geometry,
            cooling=cooling,
            mach_numbers=mach_numbers,
            T_gas_recovery=T_gas_recovery,
            h_gas=h_gas,
        )

        # Check thermal limits
        if cooling.max_wall_temp > 800:
            result.add_warning(
                f"Max wall temperature {cooling.max_wall_temp:.0f} K exceeds "
                "typical copper limit (800 K)"
            )

        self._last_result = result
        logger.info(
            f"Design complete: Isp={result.isp_vac:.1f}s, "
            f"Thrust={result.thrust_vac:.0f}N"
        )

        return result

    def analyze(
        self,
        pc_bar: Optional[float] = None,
        mr: Optional[float] = None
    ) -> EngineDesignResult:
        """
        Run off-design analysis at different operating conditions.

        Uses geometry from last design() call, recalculates performance
        at new chamber pressure and/or mixture ratio.

        Args:
            pc_bar: New chamber pressure (optional)
            mr: New mixture ratio (optional)

        Returns:
            EngineDesignResult at off-design conditions
        """
        if self._nozzle_geometry is None:
            raise ValueError("Must run design() before analyze()")

        self._init_solvers()

        pc = pc_bar or self.config.pc_bar
        mixture_ratio = mr or self.config.mr

        logger.info(f"Running off-design: Pc={pc}bar, MR={mixture_ratio}")

        # Combustion at new conditions
        combustion = self._combustion_solver.run(
            pc,
            mixture_ratio,
            self._last_result.expansion_ratio
        )

        # Mass flow at new Pc
        cstar_real = combustion.cstar * self.config.eff_combustion
        throat_area = np.pi * (self._last_result.dt_mm / 2000) ** 2
        mass_flow = (pc * 1e5) * throat_area / cstar_real

        # Gas dynamics
        mach_numbers, T_gas_recovery, h_gas = self._analyze_gas_dynamics(combustion)

        # Cooling
        cooling = self._run_cooling(mass_flow, T_gas_recovery, h_gas)

        result = EngineDesignResult(
            timestamp=datetime.now(),
            run_type="off_design",
            pc_bar=pc,
            mr=mixture_ratio,
            isp_vac=combustion.isp_vac * self.config.eff_combustion,
            isp_sea=combustion.isp_opt * self.config.eff_combustion,
            thrust_vac=mass_flow * combustion.isp_vac * self.config.eff_combustion * 9.80665,
            thrust_sea=mass_flow * combustion.isp_opt * self.config.eff_combustion * 9.80665,
            massflow_total=mass_flow,
            massflow_ox=mass_flow * mixture_ratio / (1 + mixture_ratio),
            massflow_fuel=mass_flow / (1 + mixture_ratio),
            dt_mm=self._last_result.dt_mm,
            de_mm=self._last_result.de_mm,
            length_mm=self._last_result.length_mm,
            expansion_ratio=self._last_result.expansion_ratio,
            combustion=combustion,
            nozzle_geometry=self._nozzle_geometry,
            channel_geometry=self._channel_geometry,
            cooling=cooling,
            mach_numbers=mach_numbers,
            T_gas_recovery=T_gas_recovery,
            h_gas=h_gas,
        )

        return result

    def throttle_sweep(
        self,
        min_throttle: float = 0.5,
        max_throttle: float = 1.0,
        n_points: int = 11,
        mode: str = "ox_only"
    ) -> ThrottleCurve:
        """
        Generate throttle curve across operating range.

        Args:
            min_throttle: Minimum throttle fraction (0-1)
            max_throttle: Maximum throttle fraction (0-1)
            n_points: Number of points to evaluate
            mode: Throttling mode ('ox_only', 'both', 'fuel_only')

        Returns:
            ThrottleCurve with performance at each throttle setting
        """
        if self._nozzle_geometry is None:
            raise ValueError("Must run design() before throttle_sweep()")

        throttle_levels = np.linspace(min_throttle, max_throttle, n_points)
        points: List[ThrottlePoint] = []

        for throttle in throttle_levels:
            # Calculate Pc at throttle
            pc = self.config.pc_bar * throttle

            # Adjust MR based on mode
            if mode == "ox_only":
                mr = self.config.mr * throttle
            elif mode == "fuel_only":
                mr = self.config.mr / throttle
            else:  # both
                mr = self.config.mr

            # Run analysis
            try:
                result = self.analyze(pc_bar=pc, mr=mr)
                points.append(ThrottlePoint(
                    throttle_pct=throttle * 100,
                    pc_bar=pc,
                    mr=mr,
                    thrust_n=result.thrust_sea,
                    isp_s=result.isp_sea,
                    max_wall_temp_k=result.cooling.max_wall_temp if result.cooling else 0
                ))
            except Exception as e:
                logger.warning(f"Failed at throttle={throttle}: {e}")

        return ThrottleCurve(
            points=points,
            throttle_mode=mode,
            fuel_control="venturi"
        )

    def _run_combustion(self) -> CombustionResult:
        """Run initial combustion analysis."""
        try:
            return self._combustion_solver.run(
                self.config.pc_bar,
                self.config.mr,
                10.0  # Initial estimate
            )
        except Exception as e:
            raise CombustionError(f"Combustion analysis failed: {e}")

    def _calculate_optimal_expansion(self, gamma: float) -> float:
        """Calculate optimal expansion ratio for ambient pressure."""
        from resa.physics.isentropic import get_expansion_ratio
        p_exit = self.config.p_exit_bar if self.config.p_exit_bar > 0.001 else 1.013
        return get_expansion_ratio(p_exit * 1e5, self.config.pc_bar * 1e5, gamma)

    def _size_throat(self, combustion: CombustionResult) -> tuple:
        """Size throat from thrust requirement."""
        cstar_real = combustion.cstar * self.config.eff_combustion
        isp_design = combustion.isp_opt * self.config.eff_combustion
        g0 = 9.80665

        if self.config.throat_diameter > 0.001:
            # Fixed throat
            Rt = self.config.throat_diameter / 2.0
            At = np.pi * Rt ** 2
            mass_flow = (self.config.pc_bar * 1e5) * At / cstar_real
        else:
            # Size from thrust
            mass_flow = self.config.thrust_n / (isp_design * g0)
            At = mass_flow * cstar_real / (self.config.pc_bar * 1e5)
            Rt = np.sqrt(At / np.pi)

        return Rt, mass_flow

    def _generate_nozzle(self, throat_radius: float, eps: float) -> NozzleGeometry:
        """Generate nozzle contour geometry."""
        try:
            return self._nozzle_generator.generate(
                throat_radius=throat_radius,
                expansion_ratio=eps,
                L_star_mm=self.config.L_star,
                contraction_ratio=self.config.contraction_ratio,
                theta_convergent=self.config.theta_convergent,
                bell_fraction=self.config.bell_fraction
            )
        except Exception as e:
            raise GeometryError(f"Nozzle generation failed: {e}")

    def _generate_channels(self) -> CoolingChannelGeometry:
        """Generate cooling channel geometry."""
        try:
            return self._channel_generator.generate(
                nozzle_geometry=self._nozzle_geometry,
                channel_width_throat=self.config.channel_width_throat,
                channel_height=self.config.channel_height,
                rib_width_throat=self.config.rib_width_throat,
                wall_thickness=self.config.wall_thickness,
                roughness=self.config.wall_roughness
            )
        except Exception as e:
            raise GeometryError(f"Channel generation failed: {e}")

    def _analyze_gas_dynamics(self, combustion: CombustionResult) -> tuple:
        """Analyze gas dynamics along nozzle."""
        from resa.physics.isentropic import mach_from_area_ratio
        from resa.physics.heat_transfer import calculate_adiabatic_wall_temp

        x = self._nozzle_geometry.x_full
        y = self._nozzle_geometry.y_full
        throat_idx = np.argmin(y)
        Rt = y[throat_idx]

        mach = np.zeros_like(x)
        T_recovery = np.zeros_like(x)

        gamma = combustion.gamma
        T_c = combustion.T_combustion
        Pr = 0.7  # Approximate Prandtl number

        for i, yi in enumerate(y):
            area_ratio = (yi / Rt) ** 2
            if i <= throat_idx:
                mach[i] = mach_from_area_ratio(area_ratio, gamma, supersonic=False)
            else:
                mach[i] = mach_from_area_ratio(area_ratio, gamma, supersonic=True)

            T_recovery[i] = calculate_adiabatic_wall_temp(T_c, mach[i], gamma, Pr)

        # Heat transfer coefficients
        from resa.physics.heat_transfer import calculate_bartz_coefficient
        h_gas = np.zeros_like(x)
        for i in range(len(x)):
            h_gas[i] = calculate_bartz_coefficient(
                pc=self.config.pc_bar * 1e5,
                cstar=combustion.cstar,
                Dt=Rt * 2,
                At=np.pi * Rt ** 2,
                gamma=gamma,
                mw=combustion.mw,
                T_c=T_c,
                local_area_ratio=(y[i] / Rt) ** 2,
                mach=mach[i]
            )

        return mach, T_recovery, h_gas

    def _run_cooling(
        self,
        mass_flow: float,
        T_gas: np.ndarray,
        h_gas: np.ndarray
    ) -> CoolingResult:
        """Run regenerative cooling analysis."""
        try:
            coolant_flow = mass_flow * self.config.coolant_mass_fraction
            if self.config.cooling_mode == 'counter-flow':
                # Use oxidizer fraction
                coolant_flow = mass_flow * self.config.mr / (1 + self.config.mr)
                coolant_flow *= self.config.coolant_mass_fraction

            return self._cooling_solver.solve(
                mdot_coolant=coolant_flow,
                p_in=self.config.coolant_p_in_bar * 1e5,
                t_in=self.config.coolant_t_in_k,
                geometry=self._channel_geometry,
                T_gas=T_gas,
                h_gas=h_gas,
                mode=self.config.cooling_mode
            )
        except Exception as e:
            raise CoolingError(f"Cooling analysis failed: {e}")

    @property
    def last_result(self) -> Optional[EngineDesignResult]:
        """Get the last design result."""
        return self._last_result

    @property
    def nozzle_geometry(self) -> Optional[NozzleGeometry]:
        """Get the current nozzle geometry."""
        return self._nozzle_geometry

    @property
    def channel_geometry(self) -> Optional[CoolingChannelGeometry]:
        """Get the current channel geometry."""
        return self._channel_geometry
