import cv2
from PIL import Image
import os


class RealTimeRecogniserCV:
    BASE_DIR = "faces"
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Model_LBPH.yml")
    #recognizer = cv2.face.EigenFaceRecognizer_create()
    #recognizer.read("ModelEigenFace.yml")

    def get_names(self):
        names_list = []
        for name in os.listdir(self.BASE_DIR):
            names_list.append(name)
        names_list.sort()
        return names_list

    def recognise_video_thread(self):
        video = cv2.VideoCapture(0)
        names_list = self.get_names()
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
                size = (224, 224)
                roi_gray = cv2.resize(roi_gray, size, Image.ANTIALIAS)
                id_, conf = self.recognizer.predict(roi_gray)
                print("Face:", id_, " - ", names_list[id_], conf)
                if conf<=75:
                #if conf <= 8000:
                    #print("Face:", id_, " - ", names_list[id_])
                    name = names_list[id_]
                    if name != "unknown":
                        color = (0, 255, 0)
                        font_color = (255, 255, 255)
                    else:
                        color = (0, 0, 255)
                        font_color = (0, 0, 0)
                else:
                    if conf <= 80:
                        name = "Not Shure"
                        color = (255, 0, 0)
                        font_color = (0, 0, 0)
                    else:
                        name = "unknown"
                        color = (0, 0, 255)
                        font_color = (0, 0, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # color = (255, 0, 0) #BGR 0-255
                FRAME_THICKNESS = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, FRAME_THICKNESS)
                x, y = x, end_cord_y
                end_cord_y = y + 20
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, cv2.FILLED)
                cv2.putText(frame, name, (x + 10, y + 15), font, 0.6, font_color, FRAME_THICKNESS)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        video.release()
        cv2.destroyAllWindows()


recogniser = RealTimeRecogniserCV()
recogniser.recognise_video_thread()

