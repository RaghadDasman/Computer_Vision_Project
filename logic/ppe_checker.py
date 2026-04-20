"""
logic/ppe_checker.py
--------------------
يشوف إذا العامل لابس كل معدات السلامة

منطق الموديل:
- الموديل يكشف نوعين: helmet و no_helmet
- لو كُشف no_helmet = العامل مو لابس خوذة = مخالفة
- لو كُشف helmet    = العامل لابس خوذة    = آمن
- لو ما كُشف أي منهما = نعامله كمخالفة
"""

PPE_CLASSES = {
    "person":     0,
    "helmet":     1,
    "no_helmet":  2,
    "vest":       3,
    "no_vest":    4,
    "goggles":    5,
    "no_goggles": 6,
    "gloves":     7,
    "no_gloves":  8,
}

PPE_POSITIVE = {"helmet", "vest", "goggles", "gloves"}
PPE_NEGATIVE = {
    "no_helmet":  "helmet",
    "no_vest":    "vest",
    "no_goggles": "goggles",
    "no_gloves":  "gloves",
}

REQUIRED_PPE = ["helmet", "vest", "goggles", "gloves"]

PPE_ARABIC = {
    "helmet":  "الخوذة",
    "vest":    "السترة العاكسة",
    "gloves":  "القفازات",
    "goggles": "النظارات الواقية",
}


def _boxes_overlap(box1, box2, margin: int = 40) -> bool:
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return (
        x1b - margin < x2a and
        x2b + margin > x1a and
        y1b - margin < y2a and
        y2b + margin > y1a
    )


def check_ppe(all_detections, person_bbox: list) -> dict:
    wearing     = set()
    not_wearing = set()

    for det in all_detections:
        cls_id = int(det.cls[0])
        label  = _get_label(cls_id)

        if label in ("person", "unknown"):
            continue

        ppe_box = det.xyxy[0].cpu().numpy().tolist()
        if not _boxes_overlap(person_bbox, ppe_box):
            continue

        if label in PPE_POSITIVE:
            wearing.add(label)
        elif label in PPE_NEGATIVE:
            not_wearing.add(PPE_NEGATIVE[label])

    missing = []
    for item in REQUIRED_PPE:
        if item in not_wearing or item not in wearing:
            missing.append(item)

    missing_ar = [PPE_ARABIC[p] for p in missing]

    return {
        "compliant":  len(missing) == 0,
        "present":    list(wearing),
        "missing":    missing,
        "missing_ar": missing_ar,
        "violations": list(not_wearing),
    }


def _get_label(cls_id: int) -> str:
    for name, idx in PPE_CLASSES.items():
        if idx == cls_id:
            return name
    return "unknown"
