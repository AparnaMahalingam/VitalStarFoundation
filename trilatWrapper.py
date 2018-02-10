import numpy as np
# Requires trilatDraft and trilatWrapper to be in the same folder

# Python or SQL to grab all the necessary information out of server entries
# Wraps server data i.e. ts1 = TimeStamp(server.ts, server.lat, server.lon) 
# To keep up to date, may have to reinitialize (could update, but does it save much space when everything is changing)?

class TimeStamp:
	#def __init__(self, ts, lat, lon, messageID):
	def __init__(self, ts, lat, lon):
		self.ts = ts
		self.lat = lat
		self.lon = lon
		#self.mID = messageID

	def getTimeStamp(self):
		return self.ts

	def getLatitude(self):
		return self.lat

	def getLongitude(self):
		return self.lon

	# For purposes of confirming that all the IDs match (so all received)
	# No longer needed b/c serverexplore msgNo takes its place
	# def getMessageID(self):
	# 	return self.mID

# Initialize all necessary variables (pseudo for now)
# Need some check that MsgNo is the same
# e-6 returns the worst results
# nothing makes the most sense
# ts1 = TimeStamp(14.847049	,	37.38853	,	-121.96983

# )
# ts2 = TimeStamp(14.847049	,	37.38577	,	-121.96983

# )
# ts3 = TimeStamp(14.847046	,	37.38696	,	-121.96679


# )
# tag = 1004 # to match the TapeID 1004L

# # Call the trilateration baseline (starting w/ 3 timestamps)
# # Could potentially compare with trilateration open 
# #trilatDraft.trilateration_open()
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)

# ts1 = TimeStamp(1412089132e+9	,	37.38853	,	-121.96983

# )
# ts2 = TimeStamp(1486211300e+9	,	37.38577	,	-121.96983

# )
# ts3 = TimeStamp(2810777292e+9	,	37.38696	,	-121.96679

# )
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)





# ts1 = TimeStamp(51.81821	,	37.38698	,	-121.96713

# )
# ts2 = TimeStamp(51.818207	,	37.38549	,	-121.97236


# )
# ts3 = TimeStamp(51.818212	,	37.38849	,	-121.96981


# )

# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)

# ts1 = TimeStamp(42.829207	,	37.38562	,	-121.97234

# )
# ts2 = TimeStamp(42.82921	,	37.3885	,	-121.9698



# )
# ts3 = TimeStamp(42.829203	,	37.38698	,	-121.96711



# )
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)

# ts1 = TimeStamp(16.838872	,	37.38565	,	-121.97236


# )
# ts2 = TimeStamp(16.838872	,	37.38701	,	-121.96714



# )
# ts3 = TimeStamp(16.838871	,	37.3885	,	-121.96978




# )
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)

# ts1 = TimeStamp(33.829365	,	37.38565	,	-121.97234
# )
# ts2 = TimeStamp(33.829363	,	37.38703	,	-121.96711
# )
# ts3 = TimeStamp(33.829361	,	37.3885	,	-121.96978
# )
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)

# ts1 = TimeStamp(50.83183	,	37.38703	,	-121.96711

# )
# ts2 = TimeStamp(50.831826	,	37.38565	,	-121.97234

# )
# ts3 = TimeStamp(50.831827	,	37.3885	,	-121.96978

# )
# trilatDraft.trilateration_tdoa_test_final(tag, ts1, ts2, ts3)







