import cv2
import numpy as np

class detector:
	def __init__(self):

		self.net = cv2.dnn.readNet("./model/model.weights", "./model/model.cfg")
		self.classes = []
		with open("./model/model.names", "r") as self.f:
			self.classes = [self.line.strip() for self.line in self.f.readlines()]
		self.layer_names = self.net.getLayerNames()
		self.output_layers = [self.layer_names[self.i[0] - 1] for self.i in self.net.getUnconnectedOutLayers()]

	def detect(self, image):
		self.img, self.class_ids, self.confidences, self.boxes = image, [], [], []
		self.height, self.width, self.channels = self.img.shape
		self.blob = cv2.dnn.blobFromImage(self.img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		self.net.setInput(self.blob)
		self.outs = self.net.forward(self.output_layers)

		for self.out in self.outs:
			for self.dtn in self.out:
				self.scores = self.dtn[5:]
				self.class_id = np.argmax(self.scores)
				self.confidence = self.scores[self.class_id]
				if self.confidence > 0.4:
					self.c_x, self.c_y = int(self.dtn[0] * self.width), int(self.dtn[1] * self.height)
					self.w, self.h = int(self.dtn[2]*self.width), int(self.dtn[3]*self.height)
					self.x, self.y = int(self.c_x-self.w/2), int(self.c_y-self.h/2)
					self.boxes.append([self.x, self.y, self.w, self.h])
					self.confidences.append(float(self.confidence))
					self.class_ids.append(self.class_id)

		self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
		return [self.boxes, self.indexes, self.classes, self.class_ids]
