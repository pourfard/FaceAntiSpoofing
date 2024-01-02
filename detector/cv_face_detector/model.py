from detector.FaceDetector import FaceDetectorInterface
import cv2


class CVFaceDetector(FaceDetectorInterface):
    def __init__(self):
        self.faceCascade = cv2.CascadeClassifier("detector/cv_face_detector/files/haarcascade_frontalface_default.xml")

    def get_face_bboxes(self, bgr):
        faces = self.faceCascade.detectMultiScale(bgr, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
        result = []
        for i, (x, y, w, h) in enumerate(faces):
            result.append({"box": [x, y, x + w, y + h]})
        return result
