from swirl_injector.calculators import LCSCCalculator
from swirl_injector.config import InjectorConfig

config = InjectorConfig.from_yaml("default_config.yaml")
calc = LCSCCalculator(config)
results = calc.calculate()

print(results.geometry.orifice_diameter)  # mm
print(results.performance.momentum_flux_ratio)
results.to_json("results.json")