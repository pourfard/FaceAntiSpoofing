from detector.FaceDetector import FaceDetectorInterface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from skimage import transform as trans


def crop_face(bgr, landmark):
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8
    landmark = np.asarray(landmark)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    wrapped = cv2.warpAffine(bgr, M, (112, 112), borderValue=0.0)

    return wrapped

class InsightDetector(FaceDetectorInterface):
    def __init__(self, modules=["detection"]):
        self.buff_model = FaceAnalysis(root="detector/insight_detector/files/", name="m",
                                       allowed_modules=modules,
                                       providers=["CPUExecutionProvider"])

        self.buff_model.prepare(ctx_id=0, det_size=(320, 320),
                                det_thresh=0.5)

    def get_face_bboxes(self, bgr, max_num=0):
        result = self.buff_model.get(bgr, max_num)
        x = []
        for res in result:
            x.append({"box": [int(i) for i in res.bbox], "crop": crop_face(bgr, res.kps)})
        return x
