import cv2
from typing import override
from lib.yunet import YuNet
from abc import ABC, abstractmethod
from insightface.app import FaceAnalysis
from lib.entities.face import DetectedFace


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: cv2.typing.MatLike) -> list[DetectedFace]:
        """Detect faces in the input image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list[DetectedFace]: A list of faces detected in the input image:
            [
                {
                    bbox: {
                        x: float,
                        y: float,
                        w: float,
                        h: float
                    },
                    landmarks: {
                        left_eye: (float, float),
                        right_eye: (float, float),
                        nose: (float, float),
                        left_mouth: (float, float),
                        right_mouth: (float, float)
                    } | None,
                    confidence: float
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def _convert_result_format(self, faces: list) -> list[DetectedFace]:
        """Convert the detected faces to a specific format.

        Args:
            faces: A list of faces detected in the input image.

        Returns:
            list[DetectedFace]: A list of faces detected in the input image:
            [
                {
                    bbox: {
                        x: float,
                        y: float,
                        w: float,
                        h: float
                    },
                    landmarks: {
                        left_eye: (float, float),
                        right_eye: (float, float),
                        nose: (float, float),
                        left_mouth: (float, float),
                        right_mouth: (float, float)
                    } | None,
                    confidence: float
                },
                ...
            ]
        """

    def detect_single_multiscale(
        self, image: cv2.typing.MatLike, scale_factor: float = 1.1
    ) -> tuple[DetectedFace, float] | tuple[None, None]:
        """Detect a single face in the input image with multiple scales.

        Args:
            image (MatLike): The input image.
            scale_factor (float): The factor to scale the image.

        Returns:
            out (tuple[DetectedFace, float] | tuple[None, None]): A tuple containing the detected face and the scale used to detect it. DetectedFace has the following format:
            {
                bbox: {
                    x: float,
                    y: float,
                    w: float,
                    h: float
                },
                landmarks: {
                    left_eye: (float, float),
                    right_eye: (float, float),
                    nose: (float, float),
                    left_mouth: (float, float),
                    right_mouth: (float, float)
                } | None,
                confidence: float
            }
        """

        org_h, org_w = image.shape[:2]

        scale = 1.0
        while min(scale * org_h, scale * org_w) >= 10:
            h, w = int(scale * org_h), int(scale * org_w)
            scaled_image = cv2.resize(image, (w, h))

            faces = self.detect(scaled_image)
            if len(faces) >= 2:
                return (None, None)

            if len(faces) == 1:
                return (faces[0], scale)

            scale /= scale_factor

        return (None, None)

    def visualize(
        self,
        image: cv2.typing.MatLike,
        faces: list[DetectedFace],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = False,
    ) -> cv2.typing.MatLike:
        """Visualize the detected faces on the input image.

        Args:
            image (np.ndarray): The input image.
            faces (list[list[float]]): A list of faces detected in the input image.
            color (tuple[int, int, int]): The color of the rectangle.
            thickness (int): The thickness of the rectangle.
            show_confidence (bool): Whether to show the confidence of the detected faces.

        Returns:
            np.ndarray: The image with the detected faces.
        """

        drawed_image = image.copy()
        image_h, image_w = image.shape[:2]

        for face in faces:
            x = face.bbox["x"]
            y = face.bbox["y"]
            w = face.bbox["w"]
            h = face.bbox["h"]

            # Draw rectangle around the face
            cv2.rectangle(
                drawed_image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                thickness,
            )

            # Draw confidence
            if show_confidence:
                conf = face.confidence
                cv2.putText(
                    drawed_image,
                    f"Conf: {conf:.4f}",
                    (min(image_w, max(0, int(x))), min(image_h, max(0, int(y - 10)))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    h / 500,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return drawed_image


class YuNetDetector(BaseFaceDetector):
    def __init__(
        self,
        confThreshold: float = 0.8,
    ):
        model_path = "weights/face_detection_yunet_2023mar.onnx"
        self._yunet = YuNet(model_path, confThreshold=confThreshold)

    @override
    def detect(self, image) -> list[DetectedFace]:
        h, w = image.shape[:2]
        self._yunet.setInputSize((w, h))
        faces = self._yunet.infer(image)
        return self._convert_result_format(faces)

    @override
    def _convert_result_format(self, faces: list[list[float]]) -> list[DetectedFace]:
        converted_faces = [
            DetectedFace(
                {
                    "x": face[0],
                    "y": face[1],
                    "w": face[2],
                    "h": face[3],
                },
                {
                    "left_eye": (face[4], face[5]),
                    "right_eye": (face[6], face[7]),
                    "nose": (face[8], face[9]),
                    "left_mouth": (face[10], face[11]),
                    "right_mouth": (face[12], face[13]),
                },
                face[14],
            )
            for face in faces
        ]
        return converted_faces

    def detect_single_multiscale(
        self, image, scale_factor=1.1
    ) -> tuple[DetectedFace, float] | tuple[None, None]:
        return super().detect_single_multiscale(image, scale_factor)


class RetinaFaceDetector(BaseFaceDetector):
    def __init__(self, confThreshold: float = 0.5):
        super().__init__()
        self._retinaface = FaceAnalysis(allowed_modules=["detection"])
        self._retinaface.prepare(ctx_id=0, det_thresh=confThreshold)

    @override
    def _convert_result_format(self, faces: list[dict]):
        converted_faces = [
            DetectedFace(
                {
                    "x": face["bbox"][0],
                    "y": face["bbox"][1],
                    "w": face["bbox"][2] - face["bbox"][0],
                    "h": face["bbox"][3] - face["bbox"][1],
                },
                {
                    "left_eye": face["kps"][0],
                    "right_eye": face["kps"][1],
                    "nose": face["kps"][2],
                    "left_mouth": face["kps"][3],
                    "right_mouth": face["kps"][4],
                },
                face["det_score"],
            )
            for face in faces
        ]
        return converted_faces

    @override
    def detect(self, image):
        faces = self._retinaface.get(image)
        return self._convert_result_format(faces)

    def detect_single_multiscale(
        self, image, scale_factor=1.1
    ) -> tuple[DetectedFace, float] | tuple[None, None]:
        assert (
            False
        ), "RetinaFaceDetector does not support detect_single_multiscale method."
