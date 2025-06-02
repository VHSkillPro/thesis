import cv2
import numpy as np
from abc import ABC, abstractmethod
from lib.entities.face import DetectedFace


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
        raise NotImplementedError()

    @abstractmethod
    def _convert_input_face(self, face: DetectedFace):
        """Convert the input face region to the appropriate format for the model.

        Args:
            face (DetectedFace): Face region to convert.

        Returns:
            out: The converted face region.

        """
        raise NotImplementedError()

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
