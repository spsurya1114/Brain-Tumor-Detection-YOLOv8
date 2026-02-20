def compute_tumor_metrics(box):
    """
    Compute width, height, area and severity from bounding box.
    """
    x1, y1, x2, y2 = box

    width = abs(x2 - x1)
    height = abs(y2 - y1)
    area = width * height

    # Severity classification
    if area < 2000:
        severity = "Small"
    elif area < 8000:
        severity = "Medium"
    else:
        severity = "Large"

    return width, height, area, severity