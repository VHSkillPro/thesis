import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from lib.cores.sface import SFace
from lib.entities.face import DetectedFace
from lib.face_recognizer.base import BaseFaceRecognizer


class SFaceCustomRecognizer(BaseFaceRecognizer):
    def __init__(self, model_path: str):
        super().__init__()
        self.recognizer = onnxruntime.InferenceSession(model_path)
        self.__sface = SFace("weights/face_recognition_sface_2021dec.onnx")

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
        aligned_image = self.__sface._model.alignCrop(image, converted_face)

        onnx_input = torch.from_numpy(aligned_image).permute(2, 0, 1).unsqueeze(0)
        return self.recognizer.run(None, {"input": onnx_input.float().numpy()})[0][0]
