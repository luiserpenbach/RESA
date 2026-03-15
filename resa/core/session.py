"""
Design session orchestrator for RESA.

Coordinates multi-module design workflows, caching intermediate results
and managing dependencies between analysis steps.
"""
import logging
from typing import Any, Dict, Optional

from resa.core.config import EngineConfig
from resa.core.engine import Engine
from resa.core.module_configs import (
    CoolingChannelConfig,
    FeedSystemConfig,
    PerformanceMapConfig,
    WallThicknessConfig,
)
from resa.core.results import EngineDesignResult

logger = logging.getLogger(__name__)


# Module dependency graph
MODULE_DEPENDENCIES = {
    "engine": [],
    "contour": ["engine"],
    "cooling_channels": ["engine"],
    "cooling": ["engine", "cooling_channels"],
    "wall_thickness": ["engine", "cooling"],
    "performance": ["engine"],
    "feed_system": ["engine"],
}


class DesignSession:
    """Orchestrates a multi-step engine design workflow.

    Wraps the existing Engine class and coordinates additional analysis
    modules (cooling channel design, wall thickness, performance maps,
    feed system). Caches intermediate results and tracks module status.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine = Engine(config)
        self._results: Dict[str, Any] = {}
        self._module_configs: Dict[str, Any] = {}

    @property
    def engine_result(self) -> Optional[EngineDesignResult]:
        return self._results.get("engine")

    def run_engine_design(self, with_cooling: bool = True) -> EngineDesignResult:
        """Run the core engine design pipeline."""
        logger.info("Running engine design...")
        result = self._engine.design(with_cooling=with_cooling)
        # Invalidate downstream results when engine is re-run
        self._invalidate_downstream("engine")
        self._results["engine"] = result
        return result

    def run_cooling_channels(
        self, config: CoolingChannelConfig
    ):
        """Generate cooling channel geometry with extended options."""
        self._check_dependencies("cooling_channels")
        from resa.geometry.cooling_channels import ChannelGeometryGenerator

        engine_result = self._results["engine"]
        self._module_configs["cooling_channels"] = config

        generator = ChannelGeometryGenerator()

        # Determine channel height based on profile
        if config.height_profile == "constant":
            height = config.height_throat_m
        else:
            height = config.height_throat_m  # base height, tapered later

        # Get wall thickness and other params from engine config
        geom = generator.generate(
            nozzle_geometry=engine_result.nozzle_geometry,
            channel_width_throat=self.config.channel_width_throat,
            channel_height=height,
            rib_width_throat=self.config.rib_width_throat,
            wall_thickness=self.config.wall_thickness,
            roughness=self.config.wall_roughness,
            channel_type=config.channel_type,
            taper_angle_deg=config.taper_angle_deg,
            num_channels_override=config.num_channels_override,
        )

        # Apply tapered height profile if requested
        if config.height_profile == "tapered":
            nozzle = engine_result.nozzle_geometry
            x = nozzle.x_full
            x_min, x_max = x.min(), x.max()
            x_throat = x[nozzle.y_full.argmin()]

            def tapered_height(xi):
                if xi <= x_throat:
                    frac = (xi - x_min) / (x_throat - x_min) if x_throat > x_min else 0
                    return config.height_chamber_m + frac * (
                        config.height_throat_m - config.height_chamber_m
                    )
                else:
                    frac = (xi - x_throat) / (x_max - x_throat) if x_max > x_throat else 0
                    return config.height_throat_m + frac * (
                        config.height_exit_m - config.height_throat_m
                    )

            geom = generator.generate_variable_height(geom, tapered_height)

        self._invalidate_downstream("cooling_channels")
        self._results["cooling_channels"] = geom
        return geom

    def run_cooling_analysis(self):
        """Run thermal-hydraulic cooling analysis using current channel geometry."""
        self._check_dependencies("cooling")
        engine_result = self._results["engine"]

        # Use channel geometry from cooling_channels if available,
        # otherwise fall back to engine result
        channel_geom = self._results.get("cooling_channels", engine_result.channel_geometry)

        if channel_geom is None:
            raise ValueError("No cooling channel geometry available. Run cooling_channels first.")

        from resa.solvers.cooling import RegenCoolingSolver

        solver = RegenCoolingSolver(self.config.coolant_name)
        mdot_coolant = engine_result.massflow_fuel * self.config.coolant_mass_fraction

        cooling_result = solver.solve(
            mdot_coolant=mdot_coolant,
            p_in=self.config.coolant_p_in_bar * 1e5,
            t_in=self.config.coolant_t_in_k,
            geometry=channel_geom,
            T_gas=engine_result.T_gas_recovery,
            h_gas=engine_result.h_gas,
            mode=self.config.cooling_mode,
        )

        self._invalidate_downstream("cooling")
        self._results["cooling"] = cooling_result
        return cooling_result

    def run_wall_thickness(self, config: WallThicknessConfig):
        """Run wall thickness / structural analysis."""
        self._check_dependencies("wall_thickness")
        from resa.solvers.structural import WallThicknessSolver

        self._module_configs["wall_thickness"] = config
        engine_result = self._results["engine"]
        cooling_result = self._results["cooling"]

        solver = WallThicknessSolver()
        result = solver.solve(
            nozzle_geometry=engine_result.nozzle_geometry,
            cooling_result=cooling_result,
            pc_bar=engine_result.pc_bar,
            material_name=config.material_name,
            safety_factor_pressure=config.safety_factor_pressure,
            safety_factor_thermal=config.safety_factor_thermal,
            actual_wall_thickness=self.config.wall_thickness,
        )

        self._results["wall_thickness"] = result
        return result

    def run_performance_map(self, config: PerformanceMapConfig):
        """Generate off-design performance maps."""
        self._check_dependencies("performance")
        from resa.solvers.performance import PerformanceMapSolver

        self._module_configs["performance"] = config
        engine_result = self._results["engine"]

        solver = PerformanceMapSolver()
        result = solver.solve(
            engine_result=engine_result,
            combustion=engine_result.combustion,
            config=config,
        )

        self._results["performance"] = result
        return result

    def run_feed_system(self, config: FeedSystemConfig):
        """Run feed system analysis."""
        self._check_dependencies("feed_system")
        from resa.solvers.feed_system import FeedSystemSolver

        self._module_configs["feed_system"] = config
        engine_result = self._results["engine"]

        solver = FeedSystemSolver()
        result = solver.solve(
            config=config,
            engine_result=engine_result,
        )

        self._results["feed_system"] = result
        return result

    def get_result(self, module: str) -> Optional[Any]:
        """Get cached result for a module."""
        return self._results.get(module)

    def get_module_status(self) -> Dict[str, str]:
        """Get status of each module.

        Returns:
            Dict mapping module name to status:
            - 'completed': Has valid results
            - 'ready': Dependencies met, can be run
            - 'locked': Dependencies not met
        """
        status = {}
        for module, deps in MODULE_DEPENDENCIES.items():
            if module in self._results:
                status[module] = "completed"
            elif all(d in self._results for d in deps):
                status[module] = "ready"
            else:
                status[module] = "locked"
        return status

    def _check_dependencies(self, module: str):
        """Raise if required dependencies have not been run."""
        deps = MODULE_DEPENDENCIES.get(module, [])
        missing = [d for d in deps if d not in self._results]
        if missing:
            raise ValueError(
                f"Module '{module}' requires: {missing}. "
                f"Run these modules first."
            )

    def _invalidate_downstream(self, module: str):
        """Remove cached results for modules downstream of the given module."""
        to_invalidate = set()
        for mod, deps in MODULE_DEPENDENCIES.items():
            if module in deps:
                to_invalidate.add(mod)

        for mod in to_invalidate:
            if mod in self._results:
                logger.info("Invalidating stale results for '%s'", mod)
                del self._results[mod]
            # Recursively invalidate further downstream
            self._invalidate_downstream(mod)
