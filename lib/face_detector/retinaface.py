import numpy as np
from lib.entities.face import DetectedFace
from lib.face_detector.base import BaseFaceDetector
from insightface.model_zoo.retinaface import RetinaFace


class RetinaFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        confThreshold: float = 0.5,
        model_file: str = "weights/det_10g.onnx",
    ):
        super().__init__()
        self._retinaface = RetinaFace(model_file)
        self._retinaface.prepare(-1, det_thresh=confThreshold, input_size=(640, 640))

    def _convert_result_format(
        self, faces: tuple[np.ndarray, np.ndarray]
    ) -> list[DetectedFace]:
        no_faces = faces[0].shape[0]
        converted_faces = []

        for i in range(no_faces):
            bbox = faces[0][i][:4]
            landmarks = faces[1][i]
            conf = faces[0][i][4]

            converted_faces.append(
                DetectedFace(
                    {
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2] - bbox[0],
                        "h": bbox[3] - bbox[1],
                    },
                    {
                        "left_eye": landmarks[0],
                        "right_eye": landmarks[1],
                        "nose": landmarks[2],
                        "left_mouth": landmarks[3],
                        "right_mouth": landmarks[4],
                    },
                    conf,
                )
            )

        return converted_faces

    def detect(self, image):
        faces = self._retinaface.detect(image)
        return self._convert_result_format(faces)

    def detect_single_multiscale(
        self, image, scale_factor=1.1
    ) -> tuple[DetectedFace, float] | tuple[None, None]:
        return super().detect_single_multiscale(image, scale_factor)
        # assert (
        #     False
        # ), "RetinaFaceDetector does not support detect_single_multiscale method."
