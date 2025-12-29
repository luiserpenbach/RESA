import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class InjectorQC:
    def __init__(self, design_target_flow, tolerance_percent=0.05):
        """
        design_target_flow: The ideal mass flow rate per element (kg/s)
        tolerance_percent: Allowable deviation (0.05 = +/- 5%)
        """
        self.target = design_target_flow
        self.tolerance = tolerance_percent

        # Calculate Specification Limits (The "Garage")
        self.usl = self.target * (1 + self.tolerance)  # Upper Spec Limit
        self.lsl = self.target * (1 - self.tolerance)  # Lower Spec Limit

        # Storage for batch data
        self.batch_data = []

    def add_element_test(self, measured_flow):
        """
        Add a test result from a single injector element or orifice.
        """
        self.batch_data.append(measured_flow)

    def calculate_cp_metrics(self):
        """
        Calculates Process Capability (Cp) and Centering (Cpk)
        """
        if len(self.batch_data) < 2:
            return None  # Not enough data

        data = np.array(self.batch_data)

        # 1. Statistics
        mu = np.mean(data)  # Actual Average Flow
        sigma = np.std(data, ddof=1)  # Actual Standard Deviation

        # 2. Process Potential (Cp) - Precision
        # Can the elements theoretically fit within tolerance?
        if sigma == 0:
            cp = 99.99
        else:
            cp = (self.usl - self.lsl) / (6 * sigma)

        # 3. Process Capability (Cpk) - Accuracy + Precision
        # Are the elements actually centered and fitting?
        if sigma == 0:
            cpk = 99.99
        else:
            cpu = (self.usl - mu) / (3 * sigma)  # Distance to Upper Limit
            cpl = (mu - self.lsl) / (3 * sigma)  # Distance to Lower Limit
            cpk = min(cpu, cpl)

        return {
            "count": len(data),
            "mean": mu,
            "sigma": sigma,
            "cp": cp,
            "cpk": cpk,
            "usl": self.usl,
            "lsl": self.lsl
        }

    def generate_report(self):
        stats = self.calculate_cp_metrics()

        if not stats:
            print("Insufficient data for analysis.")
            return

        print("\n" + "=" * 40)
        print(" ROCKET INJECTOR PLATE: STATISTICAL QC")
        print("=" * 40)
        print(f" Target Flow per Element: {self.target:.4f} kg/s")
        print(f" Tolerance: +/- {self.tolerance * 100}%")
        print(f" Elements Tested: {stats['count']}")
        print("-" * 40)
        print(f" ACTUAL MEAN:   {stats['mean']:.4f} kg/s")
        print(f" STD DEVIATION: {stats['sigma']:.5f}")
        print("-" * 40)
        print(f" Cp  (Precision): {stats['cp']:.2f}")
        print(f" Cpk (Capability): {stats['cpk']:.2f}")
        print("=" * 40)

        # Interpretation for Rocketry
        if stats['cpk'] < 1.33:
            print(" [CRITICAL WARNING] Cpk < 1.33")
            print(" Distribution is too wide or off-center.")
            print(" Risk: Localized O/F ratio shifts -> Hot spots on chamber wall.")
        else:
            print(" [PASS] Injector elements are consistent.")
            print(" Combustion should be uniform.")

    def plot_distribution(self):
        """
        Visualizes the manufacturing spread against the tolerance limits.
        """
        stats = self.calculate_cp_metrics()
        data = np.array(self.batch_data)

        # Create curve
        x = np.linspace(stats['lsl'] - (stats['sigma'] * 2), stats['usl'] + (stats['sigma'] * 2), 1000)
        y = norm.pdf(x, stats['mean'], stats['sigma'])

        plt.figure(figsize=(10, 6))

        # Limits
        plt.axvline(stats['lsl'], color='r', linestyle='--', linewidth=2, label='LSL (Min Allowable)')
        plt.axvline(stats['usl'], color='r', linestyle='--', linewidth=2, label='USL (Max Allowable)')
        plt.axvline(self.target, color='g', linestyle='-', linewidth=2, label='Target Design')

        # Data
        plt.plot(x, y, 'k-', lw=2, label='Actual Distribution')
        plt.fill_between(x, y, alpha=0.2, color='gray')
        plt.hist(data, bins=10, density=True, alpha=0.5, color='orange', label='Measured Samples')

        plt.title(f"Injector Element Consistency\nCp={stats['cp']:.2f} | Cpk={stats['cpk']:.2f}")
        plt.xlabel("Mass Flow Rate (kg/s)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# ==========================================
# SIMULATION / USAGE
# ==========================================

# 1. Setup: Define a LOX Injector Post
# Target flow: 0.15 kg/s per element
# Allowable variance: +/- 3%
qc_analyzer = InjectorQC(design_target_flow=0.150, tolerance_percent=0.03)

# 2. Simulate Manufacturing Data (e.g., 50 drilled holes)
# Case A: Good Precision (Cp high), but Drills were slightly small (Shifted Mean -> Low Cpk)
np.random.seed(42)
# True mean is 0.148 (slightly clogged/small), scatter is very low (0.001)
simulated_tests = np.random.normal(loc=0.148, scale=0.001, size=50)

for val in simulated_tests:
    qc_analyzer.add_element_test(val)

# 3. Run Report
qc_analyzer.generate_report()
qc_analyzer.plot_distribution()