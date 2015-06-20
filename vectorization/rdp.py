# -*- coding: utf-8 -*-

"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
"""

from math import sqrt

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
	
	
	

if __name__ == "__main__":	
	import matplotlib.pyplot as plt
	import numpy as np
	import os
	import rdp

	def angle(dir):
		"""
		Returns the angles between vectors.

		Parameters:
		dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

		The return value is a 1D-array of values of shape (N-1,), with each value
		between 0 and pi.

		0 implies the vectors point in the same direction
		pi/2 implies the vectors are orthogonal
		pi implies the vectors point in opposite directions
		"""
		dir2 = dir[1:]
		dir1 = dir[:-1]
		return np.arccos((dir1*dir2).sum(axis=1)/(
			np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))

	tolerance = 70
	min_angle = np.pi*0.22
	filename = os.path.expanduser('/media/shih/新增磁碟區/ZiWen_packup/drivers/drivers/1/1.csv')
	points = np.genfromtxt(filename).T
	print(len(points))
	x, y = points.T

	# Use the Ramer-Douglas-Peucker algorithm to simplify the path
	# http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
	# Python implementation: https://github.com/sebleier/RDP/
	simplified = np.array(rdp.rdp(points.tolist(), tolerance))

	print(len(simplified))
	sx, sy = simplified.T

	# compute the direction vectors on the simplified curve
	directions = np.diff(simplified, axis=0)
	theta = angle(directions)
	# Select the index of the points with the greatest theta
	# Large theta is associated with greatest change in direction.
	idx = np.where(theta>min_angle)[0]+1

	fig = plt.figure()
	ax =fig.add_subplot(111)

	ax.plot(x, y, 'b-', label='original path')
	ax.plot(sx, sy, 'g--', label='simplified path')
	ax.plot(sx[idx], sy[idx], 'ro', markersize = 10, label='turning points')
	ax.invert_yaxis()
	plt.legend(loc='best')
	plt.show()

	