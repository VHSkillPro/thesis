import cv2, io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from lib.entities.face import DetectedFace

plt.ioff()


def visualize_match(
    image1: cv2.typing.MatLike,
    face1: DetectedFace,
    image2: cv2.typing.MatLike,
    face2: DetectedFace,
    similarity: float,
    is_match: bool,
    output_path: str = None,
):
    drawed_image1 = image1.copy()
    drawed_image2 = image2.copy()

    x1 = face1.bbox["x"]
    y1 = face1.bbox["y"]
    w1 = face1.bbox["w"]
    h1 = face1.bbox["h"]

    x2 = face2.bbox["x"]
    y2 = face2.bbox["y"]
    w2 = face2.bbox["w"]
    h2 = face2.bbox["h"]

    color = (0, 0, 255)
    if is_match:
        color = (0, 255, 0)

    cv2.rectangle(drawed_image1, (x1, y1), (x1 + w1, y1 + h1), color, 2)
    cv2.rectangle(drawed_image2, (x2, y2), (x2 + w2, y2 + h2), color, 2)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cv2.cvtColor(drawed_image1, cv2.COLOR_BGR2RGB))
    axs[1].imshow(cv2.cvtColor(drawed_image2, cv2.COLOR_BGR2RGB))
    for ax in axs:
        ax.axis("off")
    axs[0].set_title(f"Similarity: {similarity:.4f}")
    fig.canvas.draw()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    plt.close()
    image = Image.open(buf)
    image_data = np.array(image)

    if output_path:
        image.save(output_path)
    return image_data
