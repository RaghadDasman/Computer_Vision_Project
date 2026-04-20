"""
logic/ladder_checker.py
-----------------------
1. زاوية السلم (Hough Transform)
2. تقسيم السلم لزونات (safe / warning / danger)
3. قاعدة 3 من 4 نقاط تماس (Three-Point Contact)
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. زاوية السلم
# ─────────────────────────────────────────────
def calculate_ladder_angle(frame: np.ndarray, ladder_bbox: list) -> float | None:
    """
    يحسب زاوية ميلان السلم من الأفقي.
    الزاوية المثالية: ~75°
    """
    x1, y1, x2, y2 = [int(v) for v in ladder_bbox]
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=15
    )

    if lines is None:
        return None

    angles = []
    for line in lines:
        x1l, y1l, x2l, y2l = line[0]
        dx = abs(x2l - x1l)
        dy = abs(y2l - y1l)
        if dx == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan2(dy, dx))
        angles.append(angle)

    if not angles:
        return None

    return round(float(np.median(angles)), 1)


def angle_risk_level(angle: float | None) -> str:
    """تقييم الزاوية"""
    if angle is None:
        return "unknown"
    if angle >= 70:
        return "safe"       # ✅ مثالي 70-80°
    elif angle >= 60:
        return "warning"    # ⚠️ قريب من الخطر
    else:
        return "danger"     # ❌ خطر - السلم مايل كثير


# ─────────────────────────────────────────────
# 2. تقسيم السلم لزونات
# ─────────────────────────────────────────────
def get_ladder_zones(ladder_bbox: list) -> dict:
    """
    يقسّم السلم لثلاث زونات:
    - safe   : أسفل 40%
    - warning: وسط 40%
    - danger : أعلى 20% ← الأخطر
    """
    _, y1, _, y2 = ladder_bbox
    h = y2 - y1
    return {
        "safe":    (y1 + h * 0.6,  y2),
        "warning": (y1 + h * 0.2,  y1 + h * 0.6),
        "danger":  (y1,             y1 + h * 0.2),
    }


def get_worker_zone(person_bbox: list, ladder_bbox: list) -> str:
    """وين العامل على السلم؟"""
    worker_center_y = (person_bbox[1] + person_bbox[3]) / 2
    zones = get_ladder_zones(ladder_bbox)

    if zones["danger"][0] <= worker_center_y <= zones["danger"][1]:
        return "danger"
    elif zones["warning"][0] <= worker_center_y <= zones["warning"][1]:
        return "warning"
    else:
        return "safe"


# ─────────────────────────────────────────────
# 3. قاعدة 3 من 4 (Three-Point Contact)
# ─────────────────────────────────────────────
def check_three_point_contact(keypoints: np.ndarray, ladder_bbox: list) -> dict:
    """
    keypoints : numpy array شكله (17, 3) → [x, y, confidence]
                من yolov8s-pose.pt

    Keypoint indices (COCO format):
        9  = left_wrist
        10 = right_wrist
        15 = left_ankle
        16 = right_ankle

    Returns:
        safe          : bool (True = 3 أو أكثر يلمسون السلم)
        contact_count : عدد النقاط الملامسة
        detail        : dict توضيحي
    """
    lx1, ly1, lx2, ly2 = ladder_bbox
    margin = 35  # pixel tolerance

    points_to_check = {
        "left_wrist":  9,
        "right_wrist": 10,
        "left_ankle":  15,
        "right_ankle": 16,
    }

    detail  = {}
    touching = 0

    for name, idx in points_to_check.items():
        if idx >= len(keypoints):
            detail[name] = False
            continue

        kx, ky, conf = keypoints[idx]

        if conf < 0.4:      # ثقة منخفضة = تجاهل
            detail[name] = False
            continue

        in_x = (lx1 - margin) <= kx <= (lx2 + margin)
        in_y = (ly1 - margin) <= ky <= (ly2 + margin)

        touching_point = in_x and in_y
        detail[name]   = touching_point

        if touching_point:
            touching += 1

    return {
        "safe":          touching >= 3,
        "contact_count": touching,
        "detail":        detail,
        "label":         f"{touching}/4",
    }


# ─────────────────────────────────────────────
# دالة شاملة تجمع كل شيء
# ─────────────────────────────────────────────
def full_ladder_check(
    frame:       np.ndarray,
    person_bbox: list,
    ladder_bbox: list,
    keypoints:   np.ndarray | None
) -> dict:
    """
    يرجع تقييم كامل للسلامة على السلم
    """
    angle      = calculate_ladder_angle(frame, ladder_bbox)
    angle_risk = angle_risk_level(angle)
    zone       = get_worker_zone(person_bbox, ladder_bbox)

    three_point = None
    if keypoints is not None:
        three_point = check_three_point_contact(keypoints, ladder_bbox)

    # ─── اجمع التنبيهات ───
    alerts = []

    if angle_risk == "danger":
        alerts.append(f"⚠️ زاوية السلم خطيرة ({angle}°) - الحد الأدنى 60°")
    elif angle_risk == "warning":
        alerts.append(f"⚠️ زاوية السلم قريبة من الخطر ({angle}°)")

    if zone == "danger":
        alerts.append("🚨 العامل في الزون الخطر (أعلى السلم)")
    elif zone == "warning":
        alerts.append("⚠️ العامل في منتصف السلم - تنبّه")

    if three_point is not None and not three_point["safe"]:
        count = three_point["contact_count"]
        alerts.append(f"🚨 قاعدة 3 من 4 منتهكة ({count}/4 نقاط ملامسة)")

    return {
        "angle":       angle,
        "angle_risk":  angle_risk,
        "zone":        zone,
        "three_point": three_point,
        "alerts":      alerts,
        "has_alert":   len(alerts) > 0,
    }
