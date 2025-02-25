import cv2
import glob
import numpy as np
import os.path as osp
from typing import override
from abc import ABC, abstractmethod
from insightface.app.common import Face
from lib.entities.face import DetectedFace
from insightface.model_zoo import get_model
from insightface.utils import ensure_available, face_align

from lib.sface import SFace


class BaseFaceRecognizer(ABC):
    @abstractmethod
    def infer(self, image: cv2.typing.MatLike, face: DetectedFace) -> np.ndarray:
        """Extract features from the input image.

        Args:
            image (np.ndarray): The input image.
            face (DetectedFace): Face region to extract features from.

        Returns:
            np.ndarray: The extracted features.

        """
        pass

    @abstractmethod
    def _convert_input_face(self, face: DetectedFace):
        """Convert the input face region to the appropriate format for the model.

        Args:
            face (DetectedFace): Face region to convert.

        Returns:
            out: The converted face region.

        """
        pass

    def match(
        self,
        image1: cv2.typing.MatLike,
        image2: cv2.typing.MatLike,
        face1: DetectedFace,
        face2: DetectedFace,
    ) -> float:
        """Match the similarity between two face regions in two images.

        Args:
            image1 (np.ndarray): The first input image.
            image2 (np.ndarray): The second input image.
            face1 (DetectedFace): Face region in the first image.
            face2 (DetectedFace): Face region in the second image.

        Returns:
            float: The similarity score.

        """
        features1 = self.infer(image1, face1)
        features2 = self.infer(image2, face2)
        return self.similarity(features1, features2)

    def similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Match the similarity between two feature vectors.

        Args:
            features1 (np.ndarray): The first feature vector.
            features2 (np.ndarray): The second feature vector.

        Returns:
            float: The similarity score.

        """
        from numpy.linalg import norm

        feat1 = features1.ravel()
        feat2 = features2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim


class SFaceRecognizer(BaseFaceRecognizer):
    def __init__(self):
        self.recognizer = SFace("weights/face_recognition_sface_2021dec.onnx")

    @override
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

    @override
    def infer(self, image: cv2.typing.MatLike, face: DetectedFace) -> np.ndarray:
        converted_face = self._convert_input_face(face)
        features = self.recognizer.infer(image, converted_face)
        return features


class ArcFaceRecognizer(BaseFaceRecognizer):
    def __init__(self):
        model_dir = ensure_available("models", "buffalo_l")
        onnx_files = glob.glob(osp.join(model_dir, "*.onnx"))

        onnx_file = None
        for file in onnx_files:
            if "w600k_r50" in file:
                onnx_file = file
                break

        self.recognizer = get_model(onnx_file)

    @override
    def _convert_input_face(self, face: DetectedFace):
        converted_face = Face(
            bbox=[
                face.bbox["x"],
                face.bbox["y"],
                face.bbox["w"] + face.bbox["x"],
                face.bbox["h"] + face.bbox["y"],
            ],
            kps=[
                face.landmarks["left_eye"],
                face.landmarks["right_eye"],
                face.landmarks["nose"],
                face.landmarks["left_mouth"],
                face.landmarks["right_mouth"],
            ],
            det_score=face.confidence,
        )
        return converted_face

    @override
    def infer(self, image: cv2.typing.MatLike, face: DetectedFace) -> np.ndarray:
        converted_face = self._convert_input_face(face)

        h, w = image.shape[:2]
        _x1 = int(max(0, converted_face.bbox[0]))
        _y1 = int(max(0, converted_face.bbox[1]))
        _x2 = int(min(w, converted_face.bbox[2]))
        _y2 = int(min(h, converted_face.bbox[3]))

        img_face = image[_y1:_y2, _x1:_x2]
        features = self.recognizer.get_feat(img_face)
        return features
