"""
Custom exceptions for RESA.
Provides meaningful error messages and proper error hierarchy.
"""


class RESAError(Exception):
    """Base exception for all RESA errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(RESAError):
    """Invalid or incomplete configuration."""
    pass


class ConvergenceError(RESAError):
    """Numerical solver failed to converge."""
    
    def __init__(self, message: str, iterations: int = None, residual: float = None):
        super().__init__(message, {
            'iterations': iterations,
            'residual': residual
        })
        self.iterations = iterations
        self.residual = residual


class ThermodynamicError(RESAError):
    """
    Error in thermodynamic property calculation.
    Usually from CoolProp failing at extreme conditions.
    """
    
    def __init__(self, message: str, fluid: str = None, 
                 pressure: float = None, temperature: float = None):
        super().__init__(message, {
            'fluid': fluid,
            'pressure': pressure,
            'temperature': temperature
        })
        self.fluid = fluid
        self.pressure = pressure
        self.temperature = temperature


class GeometryError(RESAError):
    """Invalid or impossible geometry."""
    pass


class CombustionError(RESAError):
    """
    Error in combustion calculation.
    Usually from CEA failing or returning invalid results.
    """
    pass


class CoolingError(RESAError):
    """Error in cooling analysis."""
    
    def __init__(self, message: str, station: int = None, 
                 temperature: float = None, pressure: float = None):
        super().__init__(message, {
            'station': station,
            'temperature': temperature,
            'pressure': pressure
        })


class FlowModelError(RESAError):
    """Error in flow model calculation (injectors, orifices, etc.)."""
    pass


class MaterialLimitError(RESAError):
    """Operating condition exceeds material limits."""
    
    def __init__(self, message: str, limit_type: str = None,
                 actual_value: float = None, limit_value: float = None):
        super().__init__(message, {
            'limit_type': limit_type,
            'actual': actual_value,
            'limit': limit_value
        })


# =============================================================================
# WARNING CLASSES (Not Exceptions)
# =============================================================================

class RESAWarning:
    """Base class for warnings (non-fatal issues)."""
    
    def __init__(self, message: str, category: str = "general"):
        self.message = message
        self.category = category
    
    def __str__(self):
        return f"[{self.category.upper()}] {self.message}"


class PerformanceWarning(RESAWarning):
    """Warning about suboptimal performance."""
    
    def __init__(self, message: str):
        super().__init__(message, "performance")


class StabilityWarning(RESAWarning):
    """Warning about potential instability."""
    
    def __init__(self, message: str):
        super().__init__(message, "stability")


class ThermalWarning(RESAWarning):
    """Warning about thermal conditions."""
    
    def __init__(self, message: str):
        super().__init__(message, "thermal")
