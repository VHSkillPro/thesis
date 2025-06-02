from lib.cores.yunet import YuNet
from lib.entities.face import DetectedFace
from lib.face_detector.base import BaseFaceDetector


class YuNetDetector(BaseFaceDetector):
    def __init__(
        self,
        confThreshold: float = 0.8,
    ):
        model_path = "weights/face_detection_yunet_2023mar.onnx"
        self._yunet = YuNet(model_path, confThreshold=confThreshold)

    def detect(self, image) -> list[DetectedFace]:
        h, w = image.shape[:2]
        self._yunet.setInputSize((w, h))
        faces = self._yunet.infer(image)
        return self._convert_result_format(faces)

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
