class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]

		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		# footage frame numbers at which each centroid was seen, used to
		# tell a parked vehicle (never moves) from a queued one
		self.frames = []
		self.parked = False
