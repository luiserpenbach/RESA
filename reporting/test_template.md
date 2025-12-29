# Rocket Engine Static Fire Report
**Test ID:** {{ test_id }}
**Date:** {{ date }}
**Engineer:** {{ engineer_name }}

## 1. Test Objectives
* Verify injector pressure drop.
* Characterize steady-state thrust at {{ target_pressure }} bar.
* **Engineer Notes:** {{ user_notes }}

## 2. Performance Summary
| Metric | Value | Unit |
| :--- | :--- | :--- |
| Max Chamber Pressure | {{ max_pc }} | bar |
| Max Thrust | {{ max_thrust }} | N |
| Burn Duration | {{ burn_time }} | s |
| Specific Impulse (est) | {{ isp }} | s |

## 3. Data Analysis
### Chamber Pressure vs. Thrust
The engine achieved steady state at T+0.5s.

![Pressure Plot]({{ plot_pressure_path }})

### Propellant Feed Pressures
Feed system behavior remained nominal throughout the burn.

![Tank Plot]({{ plot_tank_path }})

---
*Generated automatically by Python Reporting Tool*