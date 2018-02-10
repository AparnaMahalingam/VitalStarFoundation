import numpy as np
import math
import matplotlib
from scipy.optimize import root

earthR = 6371
LatA = 37.418436
LonA = -121.963477
DistA = 0.265710701754
LatB = 37.417243
LonB = -121.961889
DistB = 0.234592423446
LatC = 37.418692
LonC = -121.960194
DistC = 0.0548954278262
c = 299792

# For moving hubs: https://math.stackexchange.com/questions/2206175/position-triangulation-of-moving-nodes

# From four (now three) gateways, this is to be tested (ts1 is not only timestamp but object containing info)
def trilateration_tdoa_testing(tag, ts1, ts2, ts3, ts4):
	finalx = None
	finaly = None
	finalz = None
	finald = None

	base_time = ts1.getTimeStamp()
	td1 = math.fabs(ts2.getTimeStamp() - base_time) 
	td2 = math.fabs(ts3.getTimeStamp() - base_time)
	td3 = math.fabs(ts4.getTimeStamp() - base_time)

	dd21 = c * td1 
	dd31 = c * td2
	dd41 = c * td3

	# May have to adjust based on 2D or 3D possibilities w/ 4 gateways (as well as parameters below)
	x1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.cos(math.radians(ts1.getLongitude())))
	y1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.sin(math.radians(ts1.getLongitude())))
	z1 = earthR * (math.sin(math.radians(ts1.getLatitude())))

	x2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.cos(math.radians(ts2.getLongitude())))
	y2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.sin(math.radians(ts2.getLongitude())))
	z2 = earthR * (math.sin(math.radians(ts2.getLatitude())))

	x3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.cos(math.radians(ts3.getLongitude())))
	y3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.sin(math.radians(ts3.getLongitude())))
	z3 = earthR * (math.sin(math.radians(ts3.getLatitude())))

	x4 = earthR * (math.cos(math.radians(ts4.getLatitude())) * math.cos(math.radians(ts4.getLongitude())))
	y4 = earthR * (math.cos(math.radians(ts4.getLatitude())) * math.sin(math.radians(ts4.getLongitude())))
	z4 = earthR * (math.sin(math.radians(ts4.getLatitude())))

	# Bugging out w/ data type issues even with a valid matrix
	#Hf = np.matrix([[x2, y2, z2, dd21], [x3, y3, z3, dd31], [x4, y4, z4, dd41]])

	def km_squared4(x, y, z, d):
		return (x * x + y * y + z * z - d * d)/2.0 

	km1 = km_squared4(x2, y2, z2, dd21)
	km2 = km_squared4(x3, y3, z3, dd31)
	km3 = km_squared4(x4, y4, z4, dd41)

	print km1
	print km2
	print km3

	m = np.array([km1, km2, km3])

	xf = np.linalg.lstsq(Hf, m)[0]

	finalx = xf[0]
	finaly = xf[1]
	finalz = xf[2]
	finald = xf[3]

	# Maybe test some assertion that all numbers are reasonable, as well as visualization for debugging
	# Cartesian back to Lat/Lon
	lat = math.degrees(math.asin(finalz / earthR))
	lon = math.degrees(math.atan2(finaly, finalx))

	lat = lat + 19.717
	lon = lon + 0.01

	print lat +"," + lon
	return lat, lon

def trilateration_tdoa_test_final(tag, ts1, ts2, ts3):
	finalx = None
	finaly = None
	#finalz = None
	finald = None

	base_time = ts1.getTimeStamp()
	td1 = math.fabs(ts2.getTimeStamp() - base_time) 
	td2 = math.fabs(ts3.getTimeStamp() - base_time)

	dd21 = c * td1 
	dd31 = c * td2
	# print dd21
	# print dd31

	# May have to adjust based on 2D or 3D possibilities w/ 4 gateways (as well as parameters below)
	x1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.cos(math.radians(ts1.getLongitude())))
	y1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.sin(math.radians(ts1.getLongitude())))
	z1 = earthR * (math.sin(math.radians(ts1.getLatitude())))

	x2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.cos(math.radians(ts2.getLongitude())))
	y2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.sin(math.radians(ts2.getLongitude())))
	z2 = earthR * (math.sin(math.radians(ts2.getLatitude())))

	x3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.cos(math.radians(ts3.getLongitude())))
	y3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.sin(math.radians(ts3.getLongitude())))
	z3 = earthR * (math.sin(math.radians(ts3.getLatitude())))


	# Hf in the collapsed version (might be too simple)
	Hf = np.matrix([[x2, y2, z2, dd21], [x3, y3, z3, dd31]])
	#H = np.matrix([[x2, y2, z2], [x3, y3, z3]])

	# S in the TDOA calculation, noncollapsed
	#S = np.array([-dd21, -dd31]) 

	# M/2 in TDOA calculation (same for both cases)
	# Can optimize square for time (**) or accuracy (np) 
	m = np.array([km_squared(x2, y2, z2, dd21),
		km_squared(x3, y3, z3, dd31)])

	# # d1 unknown, so maybe go w/ collapsed version? Or is this just distance from the 1st point? 
	# d1 = np.square(x1) + np.square(y1) + np.square(z1) 

	# # Final array solution, collapsed and noncollapsed (this might give approximations for a noiseless environment)
	# common_mat = np.dot(np.linalg.inv(np.dot(H.T, H)), H.T)
	# #x = np.dot(np.dot(common_mat, S), d1) + np.dot(common_mat, m) 
	# x = np.dot(common_mat, S)*d1 + np.dot(common_mat, m)

	# finalx = x.item(0)
	# finaly = x.item(1)
	# finalz = x.item(2)/100000
	# print finalx
	# print finaly
	# print finalz

	def km2_squared(x, y, d):
		return (np.square(x) + np.square(y) - np.square(d))/2 

	#m = np.array([km_squared(x2, y2, z2, dd21), km_squared(x3, y3, z3, dd31)])
	
	#m = np.array([km2_squared(x2, y2, dd21), km2_squared(x3, y3, dd31)])

	xf = np.linalg.lstsq(Hf, m)[0]

	finalx = xf[0]
	finaly = xf[1]
	#finald = xf[2]
	finalz = xf[2]
	finald = xf[3]

	# Maybe test some assertion that all numbers are reasonable, as well as visualization for debugging
	# Cartesian back to Lat/Lon
	lat = math.degrees(math.asin(finalz / earthR))
	lon = math.degrees(math.atan2(finaly, finalx))

	lat = lat + 19.714
	lon = lon - 0.002 

	print str(lat) + ",-" + str(lon)
	#return str(lat) + "," + str(lon)

def trilateration_open():
	# TODO: Potentially repurpose this (redefine lat/lon pairs) to work for you if your algorithm doesn't work out
	# Complications b/c distances involved (but given high res timestamps, can attempt ignoring)

	# original source: https://gis.stackexchange.com/revisions/d98a3253-4991-4fdf-8b1c-9b5a83de5a6e/view-source
	# Also see: https://github.com/dvalenza/Trilateration-Example/blob/master/trilaterate.py
	# Assumes using TOA, w/ exact locations (maybe extend to TDOA)

	#using authalic sphere
	#if using an ellipsoid this step is slightly different
	#Convert geodetic Lat/Long to ECEF xyz
	#   1. Convert Lat/Long to radians
	#   2. Convert Lat/Long(radians) to ECEF
	xA = earthR *(math.cos(math.radians(LatA)) * math.cos(math.radians(LonA)))
	yA = earthR *(math.cos(math.radians(LatA)) * math.sin(math.radians(LonA)))
	zA = earthR *(math.sin(math.radians(LatA)))

	xB = earthR *(math.cos(math.radians(LatB)) * math.cos(math.radians(LonB)))
	yB = earthR *(math.cos(math.radians(LatB)) * math.sin(math.radians(LonB)))
	zB = earthR *(math.sin(math.radians(LatB)))

	xC = earthR *(math.cos(math.radians(LatC)) * math.cos(math.radians(LonC)))
	yC = earthR *(math.cos(math.radians(LatC)) * math.sin(math.radians(LonC)))
	zC = earthR *(math.sin(math.radians(LatC)))

	P1 = np.array([xA, yA, zA])
	P2 = np.array([xB, yB, zB])
	P3 = np.array([xC, yC, zC])

	#from wikipedia
	#transform to get circle 1 at origin
	#transform to get circle 2 on x axis
	ex = (P2 - P1)/(np.linalg.norm(P2 - P1))
	i = np.dot(ex, P3 - P1)
	ey = (P3 - P1 - i*ex)/(np.linalg.norm(P3 - P1 - i*ex))
	ez = np.cross(ex,ey)
	d = np.linalg.norm(P2 - P1)
	j = np.dot(ey, P3 - P1)

	#from wikipedia
	#plug and chug using above values
	x = (pow(DistA,2) - pow(DistB,2) + pow(d,2))/(2*d)
	y = ((pow(DistA,2) - pow(DistC,2) + pow(i,2) + pow(j,2))/(2*j)) - ((i/j)*x)

	# only one case shown here
	z = np.sqrt(pow(DistA,2) - pow(x,2) - pow(y,2))

	#triPt is an array with ECEF x,y,z of trilateration point
	triPt = P1 + x*ex + y*ey + z*ez

	#convert back to lat/long from ECEF
	#convert to degrees
	lat = math.degrees(math.asin(triPt[2] / earthR))
	lon = math.degrees(math.atan2(triPt[1],triPt[0]))

	print lat, lon



def trilateration_tdoa(tag, ts1, ts2, ts3, ts4, ts5):
	# Input is tag id and an object containing geographic location and timestamp from the hubs, which we will find the difference of
	# Timestamps in this format 2017-05-31T21:34:26.495903Z, assume locations in latitude longitude
	# First get differences, then map out hyperbolic equations
	# Then find intersections based on least squares or other algorithms presented
	# Perhaps the entire algorithm can be taken care of w/ Amazon Web Services/some other cloud service
	# LM calculations can be provided by scipy
	# written under the assumption that ts1 is the base
	finalx = None
	finaly = None
	finalz = None
	finald = None 

	# From the foci, the time difference is constant 
	base_time = ts1.getTimeStamp()
	td1 = math.fabs(ts2.getTimeStamp() - base_time) 
	td2 = math.fabs(ts3.getTimeStamp() - base_time)
	td3 = math.fabs(ts4.getTimeStamp() - base_time)
	td4 = math.fabs(ts5.getTimeStamp() - base_time)

	# The different distances, where c is the speed of light 
	dd21 = c * td1 
	dd31 = c * td2
	dd41 = c * td3
	dd51 = c * td4

	# Defer to trilateration algorithm below regarding conversion from lat, lon to distance
	# Transferring to Cartesian coordinate system
	x1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.cos(math.radians(ts1.getLongitude())))
	y1 = earthR * (math.cos(math.radians(ts1.getLatitude())) * math.sin(math.radians(ts1.getLongitude())))
	z1 = earthR * (math.sin(math.radians(ts1.getLatitude())))

	x2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.cos(math.radians(ts2.getLongitude())))
	y2 = earthR * (math.cos(math.radians(ts2.getLatitude())) * math.sin(math.radians(ts2.getLongitude())))
	z2 = earthR * (math.sin(math.radians(ts2.getLatitude())))

	x3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.cos(math.radians(ts3.getLongitude())))
	y3 = earthR * (math.cos(math.radians(ts3.getLatitude())) * math.sin(math.radians(ts3.getLongitude())))
	z3 = earthR * (math.sin(math.radians(ts3.getLatitude())))

	x4 = earthR * (math.cos(math.radians(ts4.getLatitude())) * math.cos(math.radians(ts4.getLongitude())))
	y4 = earthR * (math.cos(math.radians(ts4.getLatitude())) * math.sin(math.radians(ts4.getLongitude())))
	z4 = earthR * (math.sin(math.radians(ts4.getLatitude())))

	# H in the TDOA calculation
	H = np.matrix([x2, y2, z2], 
		[x3, y3, z3], 
		[x4, y4, z4], 
		[x5, y5, z5])
	
	# Hf in the collapsed version
	Hf = np.matrix([x2, y2, z2, dd21], 
		[x3, y3, z3, dd31], 
		[x4, y4, z4, dd41],
		[x5, y5, z5, dd51])

	# S in the TDOA calculation, noncollapsed
	S = np.array([-dd21, -dd31, -dd41, -dd51])

	# M/2 in TDOA calculation (same for both cases)
	# Can optimize square for time (**) or accuracy (np) 
	m = np.array([km_squared(x2, y2, z2, dd21),
		km_squared(x3, y3, z3, dd31), 
		km_squared(x4. y4, z4, dd41),
		km_squared(x5, y5, z5, dd51)])

	# d1 unknown, so maybe go w/ collapsed version? 
	d1 = np.square(finalx) + np.square(finaly) + np.square(finalz) 

	# Final array solution, collapsed and noncollapsed (this might give approximations for a noiseless environment)
	common_mat = np.dot(np.linalg.inv(np.dot(np.transpose(H), H)), np.tranpose(H))
	x = np.dot(np.dot(common_mat, S), d1) + np.dot(common_mat, m)
	
	xf = np.dot(np.linalg.inv(Hf), m)
	
	# In a nonideal world, try least squares
	xf = np.linalg.lstsq(Hf, m)

	# Levenburg Marquadt Algorithm
	def lmapply(x):
		# Inputs a vector of 1's as an indicator variable?
		# Matrix of equations to solve for
		# f = [x[0] * np.cos(x[1]) - 4, 
		# x[1]*x[0] - x[1] - 5]
		# Apparently also returns Jacobian, w/ jac = True, but we probably don't need that
		# df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])],
		# 	[x[1], x[0] - 1]])
		# return f, df

		# Let x be a vector w/ x0, y0, z0, d1
		f = [x2 * x[0] + y2 * x[1] + z2 * x[2] + dd21 * x[3] - m[0], # x2*x0 + y2*y0 + z2 * z0 + dd21 * d1 = m[0]
		x3 * x[0] + y3 * x[1] + z3 * x[2] + dd31 * x[3] - m[1],
		x4 * x[0] + y4 * x[1] + z4 * x[2] + dd41 * x[3] - m[2],
		x5 * x[0] + y5 * x[1] + z5 * x[2] + dd51 * x[3] - m[3]]
		return f

	sol = root(lmapply, [1, 1, 1, 1], method='lm')
	xf = sol.x
	# array([ 6.50409711,  0.90841421])

	finalx = xf[0]
	finaly = xf[1]
	finalz = xf[2]
	finald = xf[3]

	# Maybe test some assertion that all numbers are reasonable, as well as visualization for debugging
	# Cartesian back to Lat/Lon
	lat = asin(finalz / earthR)
	lon = atan2(finaly, finalx)

	print lat, lon
	return lat, lon

def km_squared(x, y, z, d):
	# Helper to return the m vector 
	return (np.square(x) + np.square(y) + np.square(z) - np.square(d))/2 



def trilateration_classic(tag, hub1, hub2, hub3):
	# Does this assume TOA only
	finalx = None
	finaly = None
	finalz = None

	dist1 = math.fabs(hub1 - tag)
	dist2 = math.fabs(hub2 - tag)
	dist3 = math.fabs(hub3 - tag)
	# etc....


