import cv2
from insightface.app.common import Face
import numpy as np
from lib.entities.face import DetectedFace
from lib.face_recognizer.base import BaseFaceRecognizer
from insightface.model_zoo.arcface_onnx import ArcFaceONNX


class ArcFaceRecognizer(BaseFaceRecognizer):
    def __init__(self, model_file: str = "weights/w600k_r50.onnx"):
        self._recognizer = ArcFaceONNX(model_file)

    def _convert_input_face(self, face: DetectedFace):
        converted_face = Face(
            bbox=np.array(
                [
                    face.bbox["x"],
                    face.bbox["y"],
                    face.bbox["w"] + face.bbox["x"],
                    face.bbox["h"] + face.bbox["y"],
                ]
            ),
            kps=np.array(
                [
                    face.landmarks["left_eye"],
                    face.landmarks["right_eye"],
                    face.landmarks["nose"],
                    face.landmarks["left_mouth"],
                    face.landmarks["right_mouth"],
                ]
            ),
            det_score=face.confidence,
        )
        return converted_face

    def infer(self, image: cv2.typing.MatLike, face: DetectedFace) -> np.ndarray:
        converted_face = self._convert_input_face(face)
        # Integrated face alignment in function get (insightface/model_zoo/arcface_onnx.py)
        return self._recognizer.get(image, converted_face)
