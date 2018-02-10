class DistStamp:
	#def __init__(self, ts, lat, lon, messageID):
	def __init__(self, d, lat, lon):
		self.d = d
		self.lat = lat
		self.lon = lon
		#self.mID = messageID

	def getDist(self):
		return self.d

	def getLatitude(self):
		return self.lat

	def getLongitude(self):
		return self.lon