import cv2, center, datetime
from center import process_img
from datetime import datetime
class live_feed:
	def __init__(self):

		self.o_c = process_img()

	def feed(self):

		self.cam1, self.cam2 = cv2.VideoCapture("1.mp4"), cv2.VideoCapture("2.mp4")
		self.cam3, self.cam4 = cv2.VideoCapture("3.mp4"), cv2.VideoCapture("4.mp4")
		self.count = 0
		while (True) :
			self.count += 1
			self.ret1, self.frame1 = self.cam1.read()
			self.ret2, self.frame2 = self.cam2.read()
			self.ret3, self.frame3 = self.cam3.read()
			self.ret4, self.frame4 = self.cam4.read()
			if self.count%4==0:
				if not False in [self.ret1, self.ret2, self.ret3, self.ret4]:
					self.la = self.img_stitch([self.frame1, self.frame2, self.frame3, self.frame4])

					# print("before ", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
					self.l = self.o_c.process(self.la)
					# print(self.count, "\n", "after  ", datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
					self.fin = cv2.resize(self.l, None, fx=0.4, fy=0.4)
					cv2.imshow("Detection ", self.fin)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				else:
					break

		self.cam1.release()
		self.cam2.release()
		self.cam3.release()
		self.cam4.release()
		cv2.destroyAllWindows()


	def img_stitch(self, frames):
		self.fc_list = frames

		self.frame_1_2 = cv2.hconcat([self.fc_list[0], self.fc_list[1]])
		self.frame_3_4 = cv2.hconcat([self.fc_list[2], self.fc_list[3]])
		self.con_frame = cv2.vconcat([self.frame_1_2, self.frame_3_4])

		return self.con_frame

if __name__ == "__main__":

	obj = live_feed()
	obj.feed()