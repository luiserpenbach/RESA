"""
Unit tests for configuration module.
"""

import unittest
import json
import tempfile
import os

import sys
sys.path.insert(0, '/home/claude')

from torch_igniter_simple.config import IgniterConfig, IgniterResults


class TestIgniterConfig(unittest.TestCase):
    """Test IgniterConfig dataclass."""
    
    def setUp(self):
        """Create baseline configuration for testing."""
        self.config = IgniterConfig(
            chamber_pressure=20e5,
            mixture_ratio=2.0,
            total_mass_flow=0.050,
            ethanol_feed_pressure=25e5,
            n2o_feed_pressure=30e5,
            ethanol_feed_temperature=298.15,
            n2o_feed_temperature=293.15,
            l_star=1.0,
            expansion_ratio=3.0,
            nozzle_type="conical",
            conical_half_angle=15.0,
            n2o_orifice_count=4,
            ethanol_orifice_count=4,
            discharge_coefficient=0.7,
            ambient_pressure=101325.0,
            name="test_igniter",
            description="Test configuration"
        )
    
    def test_mass_flow_split(self):
        """Test oxidizer and fuel mass flow calculations."""
        # O/F = 2.0, total = 0.050 kg/s
        # m_ox = 0.050 * 2/3 = 0.0333... kg/s
        # m_fuel = 0.050 * 1/3 = 0.0166... kg/s
        
        self.assertAlmostEqual(
            self.config.oxidizer_mass_flow,
            0.050 * 2.0 / 3.0,
            places=6
        )
        self.assertAlmostEqual(
            self.config.fuel_mass_flow,
            0.050 * 1.0 / 3.0,
            places=6
        )
        
        # Sum should equal total
        total = self.config.oxidizer_mass_flow + self.config.fuel_mass_flow
        self.assertAlmostEqual(total, self.config.total_mass_flow, places=10)
    
    def test_validation_positive_values(self):
        """Test validation rejects negative/zero values."""
        with self.assertRaises(ValueError):
            IgniterConfig(
                chamber_pressure=-1.0,  # Invalid
                mixture_ratio=2.0,
                total_mass_flow=0.050,
                ethanol_feed_pressure=25e5,
                n2o_feed_pressure=30e5,
                ethanol_feed_temperature=298.15,
                n2o_feed_temperature=293.15
            )
    
    def test_validation_feed_pressure(self):
        """Test validation requires feed pressure > chamber pressure."""
        with self.assertRaises(ValueError):
            IgniterConfig(
                chamber_pressure=20e5,
                mixture_ratio=2.0,
                total_mass_flow=0.050,
                ethanol_feed_pressure=15e5,  # Too low
                n2o_feed_pressure=30e5,
                ethanol_feed_temperature=298.15,
                n2o_feed_temperature=293.15
            )
    
    def test_json_serialization(self):
        """Test saving and loading from JSON."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            # Save
            self.config.save_json(filepath)
            
            # Load
            loaded_config = IgniterConfig.load_json(filepath)
            
            # Compare
            self.assertEqual(loaded_config.chamber_pressure, self.config.chamber_pressure)
            self.assertEqual(loaded_config.mixture_ratio, self.config.mixture_ratio)
            self.assertEqual(loaded_config.name, self.config.name)
            
        finally:
            os.unlink(filepath)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config_dict = self.config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['chamber_pressure'], 20e5)
        self.assertEqual(config_dict['mixture_ratio'], 2.0)
        self.assertEqual(config_dict['name'], 'test_igniter')


class TestIgniterResults(unittest.TestCase):
    """Test IgniterResults dataclass."""
    
    def setUp(self):
        """Create sample results for testing."""
        self.results = IgniterResults(
            flame_temperature=3000.0,
            c_star=1500.0,
            gamma=1.25,
            molecular_weight=25.0,
            heat_power_output=50000.0,
            chamber_diameter=0.050,
            chamber_length=0.150,
            chamber_volume=2.94524e-4,
            throat_diameter=0.015,
            throat_area=1.767e-4,
            exit_diameter=0.026,
            exit_area=5.309e-4,
            nozzle_length=0.025,
            n2o_orifice_diameter=0.002,
            ethanol_orifice_diameter=0.0015,
            n2o_injection_velocity=20.0,
            ethanol_injection_velocity=15.0,
            n2o_pressure_drop=10e5,
            ethanol_pressure_drop=5e5,
            isp_theoretical=200.0,
            thrust=98.0,
            c_star_efficiency=0.95,
            oxidizer_mass_flow=0.0333,
            fuel_mass_flow=0.0167,
            total_mass_flow=0.050,
            mixture_ratio=2.0,
            chamber_pressure=20e5
        )
    
    def test_summary_generation(self):
        """Test summary string generation."""
        summary = self.results.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("TORCH IGNITER DESIGN SUMMARY", summary)
        self.assertIn("COMBUSTION:", summary)
        self.assertIn("GEOMETRY:", summary)
        self.assertIn("PERFORMANCE:", summary)
    
    def test_json_serialization(self):
        """Test saving and loading results."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            # Save
            self.results.save_json(filepath)
            
            # Load
            loaded_results = IgniterResults.load_json(filepath)
            
            # Compare key values
            self.assertAlmostEqual(
                loaded_results.flame_temperature,
                self.results.flame_temperature
            )
            self.assertAlmostEqual(
                loaded_results.heat_power_output,
                self.results.heat_power_output
            )
            
        finally:
            os.unlink(filepath)


if __name__ == '__main__':
    unittest.main()
