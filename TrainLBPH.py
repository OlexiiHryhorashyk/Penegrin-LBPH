import cv2
import os
import numpy as np
from PIL import Image
from time import sleep

BASE_DIR = "faces"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


def get_names():
	names_list = []
	for name in os.listdir(BASE_DIR):
		names_list.append(name)
	names_list.sort()
	return names_list


lost = 0

names_list = get_names()
for name in os.listdir(BASE_DIR):
	for filename in os.listdir(f'{BASE_DIR}/{name}'):
		path = f'{BASE_DIR}/{name}/{filename}'
		image = cv2.imread(path)  # grayscale
		image = np.array(image, "uint8")
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
		if len(faces) == 0:
			faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
		if len(faces) == 0:
			lost += 1
		print(filename, " - ", len(faces))
		for (x, y, w, h) in faces:
			roi = image[y:y+h, x:x+w]
			size = (224, 224)
			roi = cv2.resize(roi, size, Image.ANTIALIAS)
			x_train.append(roi)
			y_labels.append(names_list.index(name))


print("Image loss:", lost)
print(y_labels)
for face in x_train:
	cv2.imshow('frame', face)
	sleep(1)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
recognizer.train(x_train, np.array(y_labels))
#recognizer.save("Model_LBPH.yml")