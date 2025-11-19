import os
import cv2
import pytest
from ultralytics import YOLO
from tests.utils_iou import yolo_to_xyxy, iou_xyxy

WEIGHTS_CANDIDATES = [
    "artifacts/model_best.pt",
    "weights/yolo11n.pt",
    "weights/yolov8n.pt",
]


def _load_model():
    for w in WEIGHTS_CANDIDATES:
        if os.path.exists(w):
            return YOLO(w)
    pytest.skip("No weights found in artifacts/ or weights/.")
    return None


@pytest.mark.slow
def test_model_recall_on_subset(test_images, yolo_labels, class_names):
    model = _load_model()
    conf_th = float(os.environ.get("CONF_TH", 0.25))
    iou_th = float(os.environ.get("IOU_TH", 0.5))

    hits = 0
    total = 0

    for img_path in test_images:
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        lbl_path = img_path.replace(
            os.sep + "images" + os.sep, os.sep + "labels" + os.sep
        )
        gts = yolo_labels(lbl_path)

        preds = model.predict(img, conf=conf_th, verbose=False)[0]
        if preds.boxes is None:
            continue
        boxes = preds.boxes.xyxy.cpu().numpy()
        clss = preds.boxes.cls.cpu().numpy().astype(int)

        for cid, cx, cy, w, h in gts:
            gt_xyxy = yolo_to_xyxy(cx, cy, w, h, W, H)
            total += 1
            best = 0.0
            for (bx1, by1, bx2, by2), c in zip(boxes, clss):
                if c != cid:
                    continue
                iou = iou_xyxy((int(bx1), int(by1), int(bx2), int(by2)), gt_xyxy)
                if iou > best:
                    best = iou
            if best >= iou_th:
                hits += 1

    recall = hits / total if total > 0 else 0.0
    print(f"Recall@IoU>={iou_th}: {recall:.3f} ({hits}/{total})")
    assert recall >= float(os.environ.get("MIN_RECALL", 0.10))
