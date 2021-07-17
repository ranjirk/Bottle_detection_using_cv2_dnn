import cv2, detection
from detection import detector

class process_img:
	def __init__(self):

		self.detObj = detector()

	def process(self, frame):

		self.frame__ = frame
		self.hei, self.wid =  self.frame__.shape[0], self.frame__.shape[1]

		self.boxes, self.indexes, self.classes, self.class_ids = self.detObj.detect(self.frame__)
		self.f1count, self.f2count, self.f3count, self.f4count = 0, 0, 0, 0

		for self.i in range(len(self.boxes)):
			if self.i in self.indexes:
				if str(self.classes[self.class_ids[self.i]]) == "bottle":
					self.x, self.y, self.w, self.h = self.boxes[self.i]

					if 0 < int( ( (self.x*2)+self.w)/2) < int(self.wid/2) :
						if 0 < int(((self.y*2)+self.h)/2) < int(self.hei/2) :
							self.f1count += 1
							self.count = self.f1count
						else :
							self.f3count += 1
							self.count = self.f3count

					elif int(self.wid/2) < int( ( (self.x*2)+self.w)/2) < self.wid :

						if 0 < int(((self.y*2)+self.h)/2) < int(self.hei/2) :

							self.f2count += 1
							self.count = self.f2count
						else :
							self.f4count += 1
							self.count = self.f4count						

					cv2.rectangle(self.frame__, (self.x, self.y), (self.x + self.w, self.y + self.h), (0,0,255), 5)
					cv2.putText(self.frame__, f"{self.count} bottle", (self.x, self.y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)

		return self.frame__
