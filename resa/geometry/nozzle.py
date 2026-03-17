"""Nozzle Geometry Generator for RESA."""
import logging

import numpy as np
from typing import Optional

from resa.core.results import NozzleGeometry

logger = logging.getLogger(__name__)


class NozzleGenerator:
    """
    Generates thrust-optimized parabolic (Rao) nozzle contours.

    Creates complete thrust chamber geometry including:
    - Cylindrical chamber section
    - Convergent cone with entrance arc
    - Throat region with upstream/downstream arcs
    - Parabolic bell nozzle expansion section
    """

    def generate(
        self,
        throat_radius: float,
        expansion_ratio: float,
        L_star_mm: float = 1000.0,
        contraction_ratio: float = 8.0,
        theta_convergent: float = 45.0,
        bell_fraction: float = 0.8,
        R_chamber: Optional[float] = None
    ) -> NozzleGeometry:
        """
        Generate complete nozzle geometry.

        Args:
            throat_radius: Throat radius [m]
            expansion_ratio: Area expansion ratio (Ae/At)
            L_star_mm: Characteristic length L* [mm]
            contraction_ratio: Area contraction ratio (Ac/At)
            theta_convergent: Convergent half-angle [degrees]
            bell_fraction: Bell nozzle length fraction (0.8 = 80% bell)
            R_chamber: Optional explicit chamber radius [m]

        Returns:
            NozzleGeometry with all contour data
        """
        Rt = throat_radius
        eps = expansion_ratio
        Re = Rt * np.sqrt(eps)

        # Determine chamber geometry
        if R_chamber is not None and R_chamber > 0:
            Rc = R_chamber
        else:
            Rc = Rt * np.sqrt(contraction_ratio)

        # Generate divergent (bell) section
        x_bell, y_bell, x_trans_div, y_trans_div, theta_n, theta_e = self._generate_rao_bell(
            Rt, Re, eps, bell_fraction
        )

        # Generate convergent (chamber) section
        x_conv, y_conv, x_cone, y_cone, x_ent, y_ent, x_cyl, y_cyl = self._generate_chamber(
            Rt, Rc, L_star_mm / 1000.0, contraction_ratio, theta_convergent
        )

        # Concatenate full contour
        x_full = np.concatenate([x_cyl, x_ent, x_cone, x_conv, x_trans_div, x_bell])
        y_full = np.concatenate([y_cyl, y_ent, y_cone, y_conv, y_trans_div, y_bell])

        # Calculate total length
        total_length = (x_full[-1] - x_full[0]) * 1000  # Convert to mm

        return NozzleGeometry(
            x_full=x_full,
            y_full=y_full,
            x_chamber=x_cyl,
            y_chamber=y_cyl,
            x_convergent=np.concatenate([x_ent, x_cone, x_conv]),
            y_convergent=np.concatenate([y_ent, y_cone, y_conv]),
            x_divergent=np.concatenate([x_trans_div, x_bell]),
            y_divergent=np.concatenate([y_trans_div, y_bell]),
            throat_radius=Rt,
            exit_radius=Re,
            chamber_radius=Rc,
            total_length=total_length,
            theta_exit=theta_e
        )

    # Rao (1958) parabolic bell nozzle angles for three reference bell fractions.
    # Source: Rao (1958) "Exhaust nozzle contour for optimum thrust"; tabulated
    # values from Huzel & Huang (1992) and NASA SP-8120 for 60%, 80%, 100% lengths.
    # θ_n: initial parabola angle at throat arc tangent point [degrees]
    # θ_e: exit angle of the parabola [degrees]
    _RAO_AR = [4, 5, 10, 20, 30, 40, 50, 100]

    _RAO_TN = {
        0.60: [25.0, 27.0, 30.5, 33.5, 35.0, 36.0, 36.5, 39.0],
        0.80: [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5],
        1.00: [18.0, 19.5, 22.5, 24.5, 25.5, 26.5, 27.0, 29.0],
    }
    _RAO_TE = {
        0.60: [18.0, 17.0, 14.0, 12.0, 11.0, 10.5, 10.0, 9.0],
        0.80: [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0],
        1.00: [10.0, 9.5, 8.5, 7.0, 6.5, 6.0, 5.5, 5.0],
    }
    _RAO_FRACTIONS = sorted(_RAO_TN.keys())  # [0.60, 0.80, 1.00]

    def _rao_angles(self, eps: float, length_fraction: float):
        """Return (θ_n, θ_e) in degrees via bilinear interpolation on ε and Lf.

        Clamps to the tabulated range [0.60, 1.00] and logs a warning for
        bell fractions outside that range.
        """
        lf = length_fraction
        if lf < self._RAO_FRACTIONS[0] or lf > self._RAO_FRACTIONS[-1]:
            logger.warning(
                "Bell fraction %.2f is outside the Rao table range [%.2f, %.2f]; "
                "clamping to boundary — angles will be approximate.",
                lf,
                self._RAO_FRACTIONS[0],
                self._RAO_FRACTIONS[-1],
            )
            lf = max(self._RAO_FRACTIONS[0], min(lf, self._RAO_FRACTIONS[-1]))

        # Find bounding fraction indices
        fracs = self._RAO_FRACTIONS
        for k in range(len(fracs) - 1):
            if fracs[k] <= lf <= fracs[k + 1]:
                f_lo, f_hi = fracs[k], fracs[k + 1]
                break
        else:
            f_lo = f_hi = fracs[-1]

        tn_lo = float(np.interp(eps, self._RAO_AR, self._RAO_TN[f_lo]))
        tn_hi = float(np.interp(eps, self._RAO_AR, self._RAO_TN[f_hi]))
        te_lo = float(np.interp(eps, self._RAO_AR, self._RAO_TE[f_lo]))
        te_hi = float(np.interp(eps, self._RAO_AR, self._RAO_TE[f_hi]))

        if f_hi == f_lo:
            return tn_lo, te_lo

        t = (lf - f_lo) / (f_hi - f_lo)
        theta_n_deg = tn_lo + t * (tn_hi - tn_lo)
        theta_e_deg = te_lo + t * (te_hi - te_lo)
        return theta_n_deg, theta_e_deg

    def _generate_rao_bell(
        self,
        Rt: float,
        Re: float,
        eps: float,
        length_fraction: float
    ):
        """Generate thrust-optimized parabolic bell nozzle."""
        # Rao angles interpolated for the given expansion ratio and bell fraction
        theta_n_deg, theta_e_deg = self._rao_angles(eps, length_fraction)
        theta_n = np.radians(theta_n_deg)
        theta_e = np.radians(theta_e_deg)

        # 15-degree conical reference length
        L_cone = (Re - Rt) / np.tan(np.radians(15))
        L_nozzle = length_fraction * L_cone

        # Throat downstream arc (0.382 * Rt radius)
        R_arc_div = 0.382 * Rt
        angle_range_trans = np.linspace(0, theta_n, 20)
        x_trans = R_arc_div * np.sin(angle_range_trans)
        y_trans = Rt + R_arc_div * (1 - np.cos(angle_range_trans))

        # Bézier control points from two-point tangent intersection
        Nx, Ny = x_trans[-1], y_trans[-1]
        Ex, Ey = L_nozzle, Re
        m1, m2 = np.tan(theta_n), np.tan(theta_e)
        C1, C2 = Ny - m1 * Nx, Ey - m2 * Ex
        denom = m1 - m2
        if abs(denom) < 1e-8:
            # Parallel tangents — degenerate case (can occur at extreme bell fractions
            # or expansion ratios). Fall back to midpoint control point.
            logger.warning(
                "Bézier control point singularity: m1≈m2=%.4f (eps=%.1f, Lf=%.2f). "
                "Using midpoint fallback.",
                m1, eps, length_fraction,
            )
            Qx = (Nx + Ex) / 2.0
            Qy = (Ny + Ey) / 2.0
        else:
            Qx = (C2 - C1) / denom
            Qy = (m1 * C2 - m2 * C1) / denom

        # Quadratic Bezier curve
        t = np.linspace(0, 1, 100)
        x_bell = ((1 - t) ** 2) * Nx + 2 * (1 - t) * t * Qx + (t ** 2) * Ex
        y_bell = ((1 - t) ** 2) * Ny + 2 * (1 - t) * t * Qy + (t ** 2) * Ey

        return x_bell, y_bell, x_trans, y_trans, theta_n_deg, theta_e_deg

    def _generate_chamber(
        self,
        Rt: float,
        Rc: float,
        L_star: float,
        CR: float,
        theta_conv_deg: float
    ):
        """Generate chamber, entrance arc, cone, and throat arc."""
        # Required chamber volume
        Vc = L_star * (np.pi * Rt ** 2)
        theta = np.radians(theta_conv_deg)

        # Throat upstream arc (1.5 * Rt radius)
        R2 = 1.5 * Rt
        alpha2 = np.linspace(theta, 0, 20)
        x_arc_throat = -R2 * np.sin(alpha2)
        y_arc_throat = Rt + R2 * (1 - np.cos(alpha2))

        xB = x_arc_throat[0]
        yB = y_arc_throat[0]

        # Entrance arc
        R1 = Rc * 1.0  # R_ent_factor = 1.0
        yA = (Rc - R1) + R1 * np.cos(theta)

        # Conical section
        dy_cone = yA - yB
        if dy_cone < 0:
            dx_cone = 0
            x_cone = np.array([])
            y_cone = np.array([])
            xA = xB
        else:
            dx_cone = dy_cone / np.tan(theta)
            xA = xB - dx_cone
            x_cone = np.linspace(xA, xB, 10, endpoint=False)
            slope = -np.tan(theta)
            y_cone = slope * (x_cone - xA) + yA

        # Entrance arc arrays
        x_center = xA - R1 * np.sin(theta)
        alpha1 = np.linspace(0, theta, 20, endpoint=False)
        x_arc_ent = x_center + R1 * np.sin(alpha1)
        y_arc_ent = (Rc - R1) + R1 * np.cos(alpha1)

        # Concatenate convergent profile for volume calculation
        x_cont = np.concatenate([x_arc_ent, x_cone, x_arc_throat])
        y_cont = np.concatenate([y_arc_ent, y_cone, y_arc_throat])

        # Volume of convergent section
        vol_conv = np.trapz(np.pi * y_cont ** 2, x_cont)

        # Cylinder length from L*
        vol_cyl_req = Vc - vol_conv
        if vol_cyl_req < 0:
            len_cyl = 0
        else:
            len_cyl = vol_cyl_req / (np.pi * Rc ** 2)

        x_cyl_end = x_arc_ent[0] if len(x_arc_ent) > 0 else xA
        x_cyl_start = x_cyl_end - len_cyl

        x_chamber = np.linspace(x_cyl_start, x_cyl_end, 50, endpoint=False)
        y_chamber = np.full_like(x_chamber, Rc)

        return (x_arc_throat, y_arc_throat,
                x_cone, y_cone,
                x_arc_ent, y_arc_ent,
                x_chamber, y_chamber)
