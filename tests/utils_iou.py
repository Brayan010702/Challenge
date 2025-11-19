def yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = max(0, int((cx - w / 2) * W))
    y1 = max(0, int((cy - h / 2) * H))
    x2 = min(W - 1, int((cx + w / 2) * W))
    y2 = min(H - 1, int((cy + h / 2) * H))
    return x1, y1, x2, y2


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    area_a = max(0, (ax2 - ax1 + 1) * (ay2 - ay1 + 1))
    area_b = max(0, (bx2 - bx1 + 1) * (by2 - by1 + 1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
