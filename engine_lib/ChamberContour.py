"""
DOC

"""


import math
import numpy as np
from bisect import bisect_left

def generate_tc_contour(aratio, Rt, l_percent, contr_ratio, L_star): # expansion_ratio, throat_radius, length percentage, contraction ratio
    # upto the nozzle designer, usually -135
    entrant_angle  	= -135
    ea_radian 		= math.radians(entrant_angle)

    # nozzle length percentage
    if l_percent == 60:		Lnp = 0.6
    elif l_percent == 80:	Lnp = 0.8
    elif l_percent == 90:	Lnp = 0.9
    else:					Lnp = 0.8
    # find wall angles (theta_n, theta_e) for given aratio (ar)
    angles = find_wall_angles(aratio, Rt, l_percent)
    # wall angles
    nozzle_length = angles[0]; theta_n = angles[1]; theta_e = angles[2];

    data_interval  	= 50

    # entrant functions
    ea_start 		= ea_radian
    ea_end 			= -math.pi/2
    angle_list1 = np.linspace(ea_start, ea_end, data_interval)
    angle_list2 = np.linspace(-ea_end , -ea_start, data_interval)
    xe = []; ye = [];

    # Chamber Section
    l = L_star / contr_ratio
    print("L_c", l)
    for i in range(0, data_interval):
      ye.append(Rt * math.sqrt(contr_ratio))
      xe.append(- (1 - i / data_interval) * l + math.sqrt(contr_ratio) * Rt * (2 * math.cos(-ea_start) + math.cos(-ea_end)))

    # Transition to throat
    for i in angle_list2:
      xe.append( - math.sqrt(contr_ratio) * Rt *  math.cos(i) + 2 * math.sqrt(contr_ratio) * Rt * math.cos(-ea_start))
      ye.append( Rt * (math.sqrt(contr_ratio) * math.sin(i)))

    # Constant gradient converging part
    for i in range(1, data_interval):
      y = math.sqrt(contr_ratio) * Rt *math.sin(-ea_start) - (i / data_interval * (math.sqrt(contr_ratio) * math.sin(-ea_start) - (2.5 * math.sin(ea_start) + 3.5)) * Rt)
      ye.append(y)
      xe.append(- y / math.tan(ea_start))

    # Throat Entrant
    for i in angle_list1:
      xe.append( 2.5 * Rt * math.cos(i))
      ye.append( 2.5 * Rt * math.sin(i) + 3.5 * Rt )

    # Throat Exit
    ea_start 		= -math.pi/2 + math.pi/720 # offset to prevent two points in nozzle throat
    ea_end 			= theta_n - math.pi/2
    angle_list3 		= np.linspace(ea_start, ea_end, data_interval)
    xe2 = []; ye2 = [];
    for i in angle_list3:
      xe2.append( 0.382 * Rt * math.cos(i) )
      ye2.append( 0.382 * Rt * math.sin(i) + 1.382 * Rt )

    # Bell Section
    # Nx, Ny-N is defined by [Eqn. 5] setting the angle to (θn – 90)
    Nx = 0.382 * Rt * math.cos(theta_n - math.pi/2)
    Ny = 0.382 * Rt * math.sin(theta_n - math.pi/2) + 1.382 * Rt
    # Ex - [Eqn. 3], and coordinate Ey - [Eqn. 2]
    Ex = Lnp * ( (math.sqrt(aratio) - 1) * Rt )/ math.tan(math.radians(15) )
    Ey = math.sqrt(aratio) * Rt
    # gradient m1,m2 - [Eqn. 8]
    m1 = math.tan(theta_n);  m2 = math.tan(theta_e);
    # intercept - [Eqn. 9]
    C1 = Ny - m1*Nx;  C2 = Ey - m2*Ex;
    # intersection of these two lines (at point Q)-[Eqn.10]
    Qx = (C2 - C1)/(m1 - m2)
    Qy = (m1*C2 - m2*C1)/(m1 - m2)

    # Selecting equally spaced divisions between 0 and 1 produces
    # the points described earlier in the graphical method
    # The bell is a quadratic Bézier curve, which has equations:
    # x(t) = (1 − t)^2 * Nx + 2(1 − t)t * Qx + t^2 * Ex, 0≤t≤1
    # y(t) = (1 − t)^2 * Ny + 2(1 − t)t * Qy + t^2 * Ey, 0≤t≤1 [Eqn. 6]
    int_list = np.linspace(0, 1, data_interval)
    xbell = []; ybell = [];
    for t in int_list:
      xbell.append( ((1-t)**2)*Nx + 2*(1-t)*t*Qx + (t**2)*Ex )
      ybell.append( ((1-t)**2)*Ny + 2*(1-t)*t*Qy + (t**2)*Ey )

    # create negative values for the other half of nozzle
    nye 	= [ -y for y in ye]
    nye2  	= [ -y for y in ye2]
    nybell  = [ -y for y in ybell]


    contour =  (xe, ye, nye, xe2, ye2, nye2, xbell, ybell, nybell)

    contourx = contour[0] # add entrant
    contourx += contour[3] # add throat exit
    contourx += contour[6] # add bell points
    contourx = np.array(contourx) # turn into numpy array

    contoury = contour[1] # same for Y
    contoury += contour[4]
    contoury += contour[7] # add bell points
    contoury = np.array(contoury) # turn into numpy array

    return contourx, contoury

# find wall angles (theta_n, theta_e) in radians for given aratio (ar)
def find_wall_angles(ar, Rt, l_percent = 80 ):

    # wall-angle empirical data
    aratio 		= [ 4,    5,    10,   20,   30,   40,   50,   100]
    theta_n_60 	= [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
    theta_n_80 	= [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
    theta_n_90 	= [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
    theta_e_60 	= [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
    theta_e_80 	= [14.0, 13.0, 11.0,  9.0,  8.5,  8.0,  7.5,  7.0]
    theta_e_90 	= [11.5, 10.5,  8.0,  7.0,  6.5,  6.0,  6.0,  6.0]

    # nozzle length calculation
    f1 = ( (math.sqrt(ar) - 1) * Rt )/ math.tan(math.radians(15) ) # 15° conical nozzle length
    if l_percent == 60:
      theta_n = theta_n_60; theta_e = theta_e_60;
      Ln = 0.6 * f1
    elif l_percent == 80:
      theta_n = theta_n_80; theta_e = theta_e_80;
      Ln = 0.8 * f1
    elif l_percent == 90:
      theta_n = theta_n_90; theta_e = theta_e_90;
      Ln = 0.9 * f1
    else:
      theta_n = theta_n_80; theta_e = theta_e_80;
      Ln = 0.8 * f1

    # find the nearest AR index in the aratio list
    x_index, x_val = find_nearest(aratio, ar)
    # if the value at the index is close to input, return it
    if round(aratio[x_index], 1) == round(ar, 1):
      return Ln, math.radians(theta_n[x_index]), math.radians(theta_e[x_index])

    # check where the index lies, and slice accordingly
    if (x_index>2):
      # slice couple of middle values for interpolation
      ar_slice = aratio[x_index-2:x_index+2]
      tn_slice = theta_n[x_index-2:x_index+2]
      te_slice = theta_e[x_index-2:x_index+2]
      # find the tn_val for given ar
      tn_val = interpolate(ar_slice, tn_slice, ar)
      te_val = interpolate(ar_slice, te_slice, ar)
    elif( (len(aratio)-x_index) <= 1):
      # slice couple of values initial for interpolation
      ar_slice = aratio[x_index-2:len(x_index)]
      tn_slice = theta_n[x_index-2:len(x_index)]
      te_slice = theta_e[x_index-2:len(x_index)]
      # find the tn_val for given ar
      tn_val = interpolate(ar_slice, tn_slice, ar)
      te_val = interpolate(ar_slice, te_slice, ar)
    else:
      # slice couple of end values for interpolation
      ar_slice = aratio[0:x_index+2]
      tn_slice = theta_n[0:x_index+2]
      te_slice = theta_e[0:x_index+2]
      # find the tn_val for given ar
      tn_val = interpolate(ar_slice, tn_slice, ar)
      te_val = interpolate(ar_slice, te_slice, ar)

    return Ln, math.radians(tn_val), math.radians(te_val)


# simple linear interpolation
def interpolate(x_list, y_list, x):
    if any(y - x <= 0 for x, y in zip(x_list, x_list[1:])):
      raise ValueError("x_list must be in strictly ascending order!")
    intervals = zip(x_list, x_list[1:], y_list, y_list[1:])
    slopes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervals]

    if x <= x_list[0]:
      return y_list[0]
    elif x >= x_list[-1]:
      return y_list[-1]
    else:
      i = bisect_left(x_list, x) - 1
      return y_list[i] + slopes[i] * (x - x_list[i])

# find the nearest index in the list for the given value
def find_nearest(array, value):
      array = np.asarray(array)
      idx = (np.abs(array - value)).argmin()
      return idx, array[idx]
