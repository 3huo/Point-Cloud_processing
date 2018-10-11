# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:03:38 2018

@author: Yanc_wang
"""

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics ):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

	z = depth_image.flatten() / 1000;
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
	"""
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud 	 	 : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image
	"""

	assert (pointcloud.shape[0] == 3)
	x_ = pointcloud[0,:]
	y_ = pointcloud[1,:]
	z_ = pointcloud[2,:]

	m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
	n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

	x = m*camera_intrinsics.fx + camera_intrinsics.ppx
	y = n*camera_intrinsics.fy + camera_intrinsics.ppy

	return x, y

def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
	"""
	Convert the depth and image point information to metric coordinates
	Parameters:
	-----------
	depth 	 	 	 : double
						   The depth value of the image point
	pixel_x 	  	 	 : double
						   The x value of the image coordinate
	pixel_y 	  	 	 : double
							The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters
	"""
	X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
	Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
	return X, Y, depth  

def get_boundary_corners_2D(points):
	"""
	Get the minimum and maximum point from the array of points
	
	Parameters:
	-----------
	points 	 	 : array
						   The array of points out of which the min and max X and Y points are needed
	
	Return:
	----------
	boundary : array
		The values arranged as [minX, maxX, minY, maxY]
	
	"""
	padding=0.05
	if points.shape[0] == 3:
		assert (len(points.shape)==2)
		minPt_3d_x = np.amin(points[0,:])
		maxPt_3d_x = np.amax(points[0,:])
		minPt_3d_y = np.amin(points[1,:])
		maxPt_3d_y = np.amax(points[1,:])

		boudary = [minPt_3d_x-padding, maxPt_3d_x+padding, minPt_3d_y-padding, maxPt_3d_y+padding]

	else:
		raise Exception("wrong dimension of points!")

	return boudary

def get_clipped_pointcloud(pointcloud, boundary):
	"""
	Get the clipped pointcloud withing the X and Y bounds specified in the boundary
	
	Parameters:
	-----------
	pointcloud 	 	 : array
						   The input pointcloud which needs to be clipped
	boundary      : array
										The X and Y bounds 
	
	Return:
	----------
	pointcloud : array
		The clipped pointcloud
	
	"""
	assert (pointcloud.shape[0]>=2)
	pointcloud = pointcloud[:,np.logical_and(pointcloud[0,:]<boundary[1], pointcloud[0,:]>boundary[0])]
	pointcloud = pointcloud[:,np.logical_and(pointcloud[1,:]<boundary[3], pointcloud[1,:]>boundary[2])]
	return pointcloud

#    Ry = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
#    pointcloud = Ry.dot(pointcloud)


def calculate_rmsd(points1, points2, validPoints=None):
	"""
	calculates the root mean square deviation between to point sets
	Parameters:
	-------
	points1, points2: numpy matrix (K, N)
	where K is the dimension of the points and N is the number of points
	validPoints: bool sequence of valid points in the point set.
	If it is left out, all points are considered valid
	"""
	assert(points1.shape == points2.shape)
	N = points1.shape[1]

	if validPoints == None:
		validPoints = [True]*N

	assert(len(validPoints) == N)

	points1 = points1[:,validPoints]
	points2 = points2[:,validPoints]

	N = points1.shape[1]

	dist = points1 - points2
	rmsd = 0
	for col in range(N):
		rmsd += np.matmul(dist[:,col].transpose(), dist[:,col]).flatten()[0]

	return np.sqrt(rmsd/N)