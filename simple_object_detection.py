import cv2, time, logging
from datetime import datetime
import numpy as np

class live_feed:
	def __init__(self):
		logging.basicConfig(filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
		logging.debug("\nLoading model...")
		self.modelStart = time.time()
		self.classes = []
		with open("model.names", "r") as self.f:
			self.classes = [self.line.strip() for self.line in self.f.readlines()]
		self.net = cv2.dnn.readNet("model.weights", "model.cfg")
		self.layer_names = self.net.getLayerNames()
		self.output_layers = [self.layer_names[self.i - 1] for self.i in self.net.getUnconnectedOutLayers()]

		logging.debug(f"Model loaded timr taken {time.time() - self.modelStart}")

		self.cam = cv2.VideoCapture("sample.mp4")
		self.count = 0
		while (True) :
			self.count += 1
			if self.count %5 == 0 :
				self.ret, self.frame = self.cam.read()
				if self.ret :
					self.frameStart = time.time()
					self.frame_1, self.resBox = self.process_img(self.frame)
					logging.debug(f"Time taken for {self.count}	{self.resBox}		{time.time() - self.frameStart} ")
					cv2.imshow("Detection ", self.frame_1)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				else:
					break
		self.cam.release()
		cv2.destroyAllWindows()


	def process_img(self, image):

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
		self.flag = False
		for self.i in range(len(self.boxes)):
			if self.i in self.indexes:
				if str(self.classes[self.class_ids[self.i]]) == "bottle":
					self.flag = True
					self.x, self.y, self.w, self.h = self.boxes[self.i]
					cv2.rectangle(self.frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (0,0,255), 5)

		return self.frame, self.flag

if __name__ == "__main__":
	obj = live_feed()