import numpy as np
import cv2

class  gun_detect():
	def __init__(self, image):
		self.name = "detect_"+str(image)
		self.conf_threshold = 0.2
		self.nms_threshold = 0.1
		self.file_label = "main/yolo.names"
		self.file_cfg = "main/yolov3.cfg"
		self.file_weight = "main/yolov3.backup"
		if image:
			self.image = cv2.imread("media/" + image)

	def predict(self, image):
		LABELS = open(self.file_label).read()
		net = cv2.dnn.readNetFromDarknet(self.file_cfg, self.file_weight)
		(H, W) = image.shape[:2]
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)
		boxes = []
		confidences = []
		classIDs = []
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > self.conf_threshold:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the image
				color = (0,255,255)
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.2f}".format(LABELS, confidences[i] * 100)
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)
			return image
			
	def save(self):
		result = self.predict(self.image)
		cv2.imwrite("media/"+self.name, result)
		return "/media/"+self.name

# if __name__ == "__main__":
# 	img = "media/armas (5).jpg"
# 	a = gun_detect(img)
# 	b = a.save()
# 	print(b)