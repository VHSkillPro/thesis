import cv2
import numpy as np
from lib.cores.sface import SFace
from lib.entities.face import DetectedFace
from lib.face_recognizer.base import BaseFaceRecognizer


class SFaceRecognizer(BaseFaceRecognizer):
    def __init__(self):
        self.recognizer = SFace("weights/face_recognition_sface_2021dec.onnx")

    def _convert_input_face(self, face: DetectedFace):
        converted_face = np.array(
            [
                face.bbox["x"],
                face.bbox["y"],
                face.bbox["w"],
                face.bbox["h"],
                *face.landmarks["left_eye"],
                *face.landmarks["right_eye"],
                *face.landmarks["nose"],
                *face.landmarks["left_mouth"],
                *face.landmarks["right_mouth"],
                face.confidence,
            ],
            dtype=np.float32,
        )
        return converted_face

    def infer(self, image: cv2.typing.MatLike, face: DetectedFace) -> np.ndarray:
        converted_face = self._convert_input_face(face)
        # Integrated face alignment in the SFace model (function _preprocess in SFace class)
        features = self.recognizer.infer(image, converted_face)
        return features
