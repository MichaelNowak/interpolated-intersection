import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def interpolated_intersection(x_1, y_1, x_2, y_2, acc=1000, spline='linear'):
	x_1_dx_avg = (np.amax(x_1) - np.amin(x_1)) / len(x_1)
	x_2_dx_avg = (np.amax(x_2) - np.amin(x_2)) / len(x_2)
	dx = np.amin((x_1_dx_avg, x_2_dx_avg))
	# interpolation resolution
	# minimal step size of input arrays is reduced
	step = dx / 2
	# concatenated arrays to determine new range min and max
	x_intersect_min = np.amax((np.amin(x_1), np.amin(x_2)))
	x_intersect_max = np.amin((np.amax(x_1), np.amax(x_2)))
	# domain for interpolation with resolution 'step'
	x_ip = np.arange(x_intersect_min, x_intersect_max, step)
	
	# create interpolation functions with given arrays
	f = interpolate.interp1d(x_1, y_1, kind=spline)
	g = interpolate.interp1d(x_2, y_2, kind=spline)
	
	#interpolation function on new interpolation domain with resoltion res
	f_ip = f(x_ip)
	g_ip = g(x_ip)

	# indices (or domain values) of the intersection points
	idx = np.argwhere(np.diff(np.sign(f_ip - g_ip))).flatten()

	# increased precision for localisation of intersection
	# new interpolation in vicinity of intersection points with accuracy 'acc'
	res_res = acc
	p = np.array([])
	q = np.array([])
	for i in idx:
		k = interpolate.interp1d([x_ip[i], x_ip[i+1]], [f_ip[i], f_ip[i+1]])
		l = interpolate.interp1d([x_ip[i], x_ip[i+1]], [g_ip[i], g_ip[i+1]])
		
		z_ip = np.linspace(x_ip[i], x_ip[i+1], res_res)
		
		k_ip = k(z_ip)
		l_ip = l(z_ip)
		
		idx_idx = np.argwhere(np.diff(np.sign(k_ip - l_ip))).flatten()
		
		p = np.append(p, [z_ip[idx_idx]])
		# k_ip and l_ip return the same solution as it is the intersection value
		q = np.append(q, [k_ip[idx_idx]])

	return np.concatenate((p, q))
	
# demonstration and testing

# two intersecting functions with different domain step sizes and domain limits
x_f = np.linspace(0, 12, 500)
f = np.sin(x_f)

x_g = np.linspace(2, 15, 900)
g = np.cos(x_g)

# applying defined function for intersection points and storing in (x, y)
x, y = np.split(interpolated_intersection(x_f, f, x_g, g), 2)

# plot functions
plt.plot(x_f, f)
plt.plot(x_g, g)

# plot intersection points
plt.plot(x, y, 'ro')

plt.show()