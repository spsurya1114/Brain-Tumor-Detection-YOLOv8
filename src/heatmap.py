import cv2
import numpy as np


def create_heatmap(image_bgr, boxes=None):
    """
    Generate attention heatmap and overlay it on MRI image.

    Parameters:
    ----------
    image_bgr : np.ndarray
        Input image in BGR format

    boxes : list (optional)
        Bounding boxes [[x1,y1,x2,y2], ...]
        If provided, heatmap focuses on tumor region
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Smooth noise
    blurred = cv2.GaussianBlur(gray, (25, 25), 0)

    # Normalize intensity
    heatmap = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to colored heatmap
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)

    # If bounding boxes exist, mask heatmap outside tumor
    if boxes:
        mask = np.zeros_like(gray, dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255

        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)

    return overlay