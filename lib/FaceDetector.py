import cv2
from abc import ABC, abstractmethod
from lib.yunet import YuNet


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: cv2.typing.MatLike) -> list[list[float]]:
        """Detect faces in the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list[list[float]]: A list of faces detected in the input image.
        """
        pass

    def detect_single_multiscale(
        self, image: cv2.typing.MatLike, scale_factor: float = 1.1
    ) -> tuple[list[float], float] | None:
        """Detect a single face in the input image with multiple scales.

        Args:
            image (np.ndarray): The input image.
            scale_factor (float): The factor to scale the image.

        Returns:
            tuple[list[float], float] | None: A tuple containing the detected face and the scale used to detect it.
        """

        org_h, org_w = image.shape[:2]

        scale = 1.0
        while min(scale * org_h, scale * org_w) >= 50:
            h, w = int(scale * org_h), int(scale * org_w)
            scaled_image = cv2.resize(image, (w, h))

            faces = self.detect(scaled_image)
            if len(faces) >= 2:
                return None, None

            if len(faces) == 1:
                return (faces[0], scale)

            scale /= scale_factor

        return None

    def visualize(
        self,
        image: cv2.typing.MatLike,
        faces: list[list[float]],
        color: tuple[int, int, int] = (0, 255, 0),
        thinckness: int = 2,
        show_confidence: bool = False,
    ) -> cv2.typing.MatLike:
        """Visualize the detected faces on the input image.

        Args:
            image (np.ndarray): The input image.
            faces (list[list[float]]): A list of faces detected in the input image.
            color (tuple[int, int, int]): The color of the rectangle.
            thinckness (int): The thickness of the rectangle.
            show_confidence (bool): Whether to show the confidence of the detected faces.

        Returns:
            np.ndarray: The image with the detected faces.
        """

        drawed_image = image.copy()

        for face in faces:
            x, y, w, h = face[:4]

            # Draw rectangle around the face
            cv2.rectangle(
                drawed_image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                thinckness,
            )

            # Draw confidence
            if show_confidence:
                conf = face[4]
                cv2.putText(
                    drawed_image,
                    f"{conf:.2f}",
                    (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    thinckness,
                    cv2.LINE_AA,
                )

        return drawed_image


class YuNetDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str = "weights/face_detection_yunet_2023mar.onnx",
        confThreshold: float = 0.8,
    ):
        self._yunet = YuNet(model_path, confThreshold=confThreshold)

    def detect(self, image) -> list[list[float]]:
        h, w = image.shape[:2]
        self._yunet.setInputSize((w, h))
        faces = self._yunet.infer(image)

        # formatted_faces = [[*face[:4], face[14]] for face in faces]
        # return formatted_faces
        return faces

    def detect_single_multiscale(
        self, image, scale_factor=1.1
    ) -> tuple[list[float], float] | None:
        return super().detect_single_multiscale(image, scale_factor)
