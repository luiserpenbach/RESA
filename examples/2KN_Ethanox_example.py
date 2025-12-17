from rocket_engine.core.thrust_chamber import EngineSpec, ThrustChamber

spec = EngineSpec(
    name="Hopper E2-C01",
    thrust=2_200,
    p_c=25,
    fuel="Ethanol90",
    oxidizer="N2O",
    l_star=1.3,
    contraction_ratio=10,
    nozzle_length_pct=0.8
)

engine = ThrustChamber(spec)
engine.analyze_performance()
engine.design_geometry()
engine.summary()
engine.plot(save=True)