from numpy import sqrt, pi

mf = 0.020           # kg/s
d = 1.0* 1e-3    # m
A = pi/4 * d**2     # m^3
rho = 1001          # kg/m^3
dp = 4.6*1e5       # Pa

cd = mf / (A*sqrt(2*rho*dp))
print(f"C_d value: {cd:.4f}")