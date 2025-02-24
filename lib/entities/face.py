from typing import TypedDict


class Bbox(TypedDict):
    x: float
    y: float
    w: float
    h: float


class Landmarks(TypedDict):
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    left_mouth: tuple[float, float]
    right_mouth: tuple[float, float]


class DetectedFace:
    def __init__(
        self, bbox: Bbox, landmarks: Landmarks | None = None, confidence: float = None
    ):
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence

    def __str__(self):
        return "DetectedFace(bbox={}, landmarks={}, confidence={})".format(
            self.bbox, self.landmarks, self.confidence
        )
