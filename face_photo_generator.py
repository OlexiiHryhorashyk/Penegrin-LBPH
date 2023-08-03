import cv2
# import os
# import numpy as np
# from PIL import Image
#
from time import sleep
#
# video = cv2.VideoCapture(0)
#
# BASE_DIR = "faces"
# name = "Olexii"
# i = 0
# while True:
# 	et, frame = video.read()
# 	filename = "me_generated"+str(i)+".jpg"
# 	path = f'{BASE_DIR}/{name}/{filename}'
# 	cv2.imwrite('path', frame)
# 	sleep(1)
# 	i+=1
# 	if cv2.waitKey(20) & 0xFF == ord('q'):
# 		break
# import cv2

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0

#BASE_DIR = "faces"
BASE_DIR = "known_faces"
name = "Olexii"

while True:
	ret, frame = cap.read()

	# This condition prevents from infinite looping
	# incase video ends.
	if not ret:
		continue
	# Save Frame by Frame into disk using imwrite method
	filename = "me_generated"+str(i)+".jpg"
	path = f'{BASE_DIR}/{name}/{filename}'
	cv2.imshow('frame', frame)
	cv2.imwrite(path, frame)
	i += 1
	sleep(0.1)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()