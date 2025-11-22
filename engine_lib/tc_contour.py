"""
TOP (Thrust Optimized Parabola) Nozzle Generation after Rao

For Reference:
[1] http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf
"""
#import ezdxf
import numpy as np
import matplotlib.pyplot as plt
import csv

from ezdxf.math import Vec2

DEG_TO_RAD = np.pi / 180  # degrees to radians factor


def get_theta(position, area_ratio, nozzle_fraction=0.8):
    # Get angle at inflection or end point of nozzle by interpolating graph by Rao for different ratios
    if nozzle_fraction == 0.8:
        # inflection point angles
        data_n = {
            "area_ratio": [3.678, 3.854, 4.037, 4.229, 4.431, 4.642, 4.863, 5.094, 5.337, 5.591, 5.857, 6.136, 6.428,
                           6.734, 7.055, 7.391, 7.743, 8.111, 8.498, 8.902, 9.326, 9.77, 10.235, 10.723, 11.233, 11.768,
                           12.328, 12.915, 13.53, 14.175, 14.85, 15.557, 16.297, 17.074, 17.886, 18.738, 19.63, 20.565,
                           21.544, 22.57, 23.645, 24.771, 25.95, 27.186, 28.48, 29.836, 31.257, 32.746, 34.305, 35.938,
                           37.649, 39.442, 41.32, 43.288, 45.349, 47.508, 49.77, 52.14, 54.623],
            "theta_n": [21.067, 21.319, 21.601, 21.908, 22.215, 22.482, 22.734, 22.986, 23.238, 23.489, 23.736, 23.984,
                        24.232, 24.48, 24.728, 24.965, 25.176, 25.387, 25.598, 25.809, 26.02, 26.231, 26.441, 26.617,
                        26.792, 26.968, 27.143, 27.319, 27.494, 27.67, 27.845, 27.996, 28.134, 28.272, 28.409, 28.547,
                        28.684, 28.822, 28.965, 29.119, 29.272, 29.426, 29.58, 29.733, 29.887, 30.04, 30.169, 30.298,
                        30.426, 30.554, 30.683, 30.811, 30.94, 31.085, 31.239, 31.393, 31.546, 31.7, 31.853]}
        # exit point angles
        data_e = {
            "area_ratio": [3.678, 3.854, 4.037, 4.229, 4.431, 4.642, 4.863, 5.094, 5.337, 5.591, 5.857, 6.136, 6.428,
                           6.734, 7.055, 7.391, 7.743, 8.111, 8.498, 8.902, 9.326, 9.77, 10.235, 10.723, 11.233, 11.768,
                           12.328, 12.915, 13.53, 14.175, 14.85, 15.557, 16.297, 17.074, 17.886, 18.738, 19.63, 20.565,
                           21.544, 22.57, 23.645, 24.771, 25.95, 27.186, 28.48, 29.836, 31.257, 32.746, 34.305, 35.938,
                           37.649, 39.442, 41.32, 43.288, 45.349, 47.508],
            "theta_e": [14.355, 14.097, 13.863, 13.624, 13.372, 13.113, 12.889, 12.684, 12.479, 12.285, 12.096, 11.907,
                        11.733, 11.561, 11.393, 11.247, 11.101, 10.966, 10.832, 10.704, 10.585, 10.466, 10.347, 10.229,
                        10.111, 10.001, 9.927, 9.854, 9.765, 9.659, 9.553, 9.447, 9.341, 9.235, 9.133, 9.047, 8.962,
                        8.877, 8.797, 8.733, 8.67, 8.602, 8.5, 8.398, 8.295, 8.252, 8.219, 8.187, 8.155, 8.068, 7.96,
                        7.851, 7.744, 7.68, 7.617, 7.553]}
    else:
        raise ValueError("No data availble for this nozzle length fraction!")

    if area_ratio < 3.7 or area_ratio > 47:
        raise ValueError("Area ratio provided is outside of the range of availble data!")
    else:
        # linear interpolation to get right angle in radians for given nozzle expansion ratio
        if position == 0:  # return for inflextion point
            return np.interp(area_ratio, data_n["area_ratio"], data_n["theta_n"]) * DEG_TO_RAD
        elif position == 1:
            return np.interp(area_ratio, data_e["area_ratio"], data_e["theta_e"]) * DEG_TO_RAD
        else:
            raise ValueError("Invalid position requested! Use 0 to get theta_n and 1 to get theta_e.")


def get_contour(R_t, area_ratio, contraction_ratio, L_star, theta_conv=45, nozzle_fraction=0.8):
    # Return x and y coordinates of the thrust chamber contour with bell nozzle

    R_c = R_t * np.sqrt(contraction_ratio)  # Chamber radius
    L_c = L_star / contraction_ratio  # Chamber section length

    try:
        theta_n = get_theta(0, area_ratio, nozzle_fraction)
        theta_e = get_theta(1, area_ratio, nozzle_fraction)
        print(theta_n * 180 / np.pi)
        print(theta_e * 180 / np.pi)
    except ValueError as e:
        print(e)

    xs = []  # Contour arrays
    ys = []

    R_e = np.sqrt(area_ratio) * R_t  # Nozzle exit radius
    resolution = 200

    # --- Chamber Transition to throat (left to right)
    for theta in np.linspace(np.pi / 2, (90 - theta_conv) * DEG_TO_RAD, resolution):  # change to (90-theta_conv)
        xs.append(R_c * np.cos(theta)
                  - abs(2.5 * R_t * np.sin((180 - theta_conv) * DEG_TO_RAD))  # THROAT ENTRANT LENGTH on X
                  - abs(R_c * np.cos((90 - theta_conv) * DEG_TO_RAD))  # CHAMBER TRANSITION LENGTH on X
                  - abs((R_c * np.sin((90 - theta_conv) * DEG_TO_RAD) - (
                    R_t + 2.5 * R_t - abs(2.5 * R_t * np.cos((180 - theta_conv) * DEG_TO_RAD)))) / np.tan(
            theta_conv * DEG_TO_RAD)))  # LINEAR SECTION LENGTH
        ys.append(R_c * np.sin(theta))

    # --- Throat Section
    # Throat Entrant Arc (top to bottom)
    for theta in np.linspace((180 - theta_conv) * DEG_TO_RAD, np.pi, resolution, endpoint=False):
        xs.append(-2.5 * R_t * np.sin(theta))
        ys.append(R_t + 2.5 * R_t + 2.5 * R_t * np.cos(theta))  # Equation 4 from [1]

    # Throat Exit Arc (top to bottom)
    for theta in np.linspace(np.pi, np.pi + theta_n, resolution):
        xs.append(-0.382 * R_t * np.sin(theta))
        ys.append(R_t + 0.382 * R_t + 0.382 * R_t * np.cos(theta))  # Equation 5 from [1]

    # --- Bell Section
    # Nozzle Start Point
    Nx = xs[-1]  # Set nozzle starting point on throat exit point
    Ny = ys[-1]

    # Nozzle End Point
    Ex = nozzle_fraction * (np.sqrt(area_ratio) - 1) * R_t / np.tan(15 * DEG_TO_RAD)  # Equation 3
    Ey = R_e

    # Point Q localization
    m1 = np.tan(theta_n)
    m2 = np.tan(theta_e)  # Equation 8
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex  # Equation 9

    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Add parabola points to contour
    for t in np.linspace(0, 1, resolution):
        xs.append((1 - t) ** 2 * Nx + 2 * (1 - t) * t * Qx + t ** 2 * Ex)
        ys.append((1 - t) ** 2 * Ny + 2 * (1 - t) * t * Qy + t ** 2 * Ey)

    # --- Combustion Chamber
    # Linear section end point
    ys.insert(0, R_c)
    dy = ys[0] - ys[1]
    dx = dy / abs(np.tan(theta_conv * DEG_TO_RAD))
    xs.insert(0, xs[0] - dx)

    # Linear Section start point
    ys.insert(0, R_c)
    xs.insert(0, xs[0] - L_c)

    return xs, ys


def write_coordinates_to_csv(x_coords, y_coords, filename):
    """
    Writes x and y coordinates to a CSV file.

    Parameters:
    x_coords (list): A list of x-coordinates.
    y_coords (list): A list of y-coordinates.
    z_coords (list): Zero List
    filename (str): The name of the output CSV file.

    Raises:
    ValueError: If the lengths of x_coords and y_coords are not the same.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")

    with open(filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header row
        csv_writer.writerow(["x", "y", "z"])

        # Write each coordinate pair
        for x, y in zip(x_coords, y_coords):
            csv_writer.writerow([x, y, 0])


def write_coordiantes_to_txt(x_coords, y_coords, filename):
    if len(x_coords) != len(y_coords):
        raise ValueError("Error: Not tthe same amount of values!")

    with open(filename, mode='w', ) as txtfile:

        txtfile.write("x\ty\n")

        for x, y in zip(x_coords, y_coords):
            txtfile.write(f"{x}\t{y}\n")

def write_coordinates_to_dxf(x_coords, y_coords, filename):
    """
    Writes x and y coordinates to a DXF file as a 2D polyline for CAD import.

    Parameters:
    x_coords (list): A list of x-coordinates.
    y_coords (list): A list of y-coordinates.
    filename (str): The name of the output DXF file.

    Raises:
    ValueError: If the lengths of x_coords and y_coords are not the same.
    """
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")

    # Create a new DXF document
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()

    # Create a polyline from the coordinates
    points = [Vec2(x, y) for x, y in zip(x_coords, y_coords)]
    polyline = msp.add_lwpolyline(points)

    # Save the DXF file
    doc.saveas(filename)


# --- Testing code
R_t = 14.35
ER = 4.0416
CR = 12
L_star = 1200
xs, ys = get_contour(R_t, ER, CR, L_star, theta_conv=30)

plt.style.use('dark_background')
figure, axis1 = plt.subplots()
plt.grid(True, color="gray", linestyle='dotted')
axis1.set_aspect('equal')
axis1.plot(xs, ys, "-", label='Contour', color="silver", linewidth=3)
axis1.plot(xs, np.dot(-1, ys), color="silver")  # mirrow chamber wall

axis1.set_xlabel('Contour X [mm]')
axis1.set_ylabel('Contour Y [mm]')
axis1.set_ylim([-ys[0] - 5, ys[0] + 5])

#write_coordinates_to_dxf(xs, ys, "HE01-inner_chamber.dxf")

#write_coordinates_to_csv(xs, ys, "../Analysis_Output_Temp/HE01-inner_chamber_contour.csv")

plt.show()
