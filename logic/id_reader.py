"""
logic/id_reader.py
------------------
يقرأ رقم الـ ID من صورة الـ badge
"""

import easyocr
import numpy as np
import re
import cv2

# أنشئ الـ reader مرة وحدة (بطيء في الأول)
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        print("🔄 Loading EasyOCR...")
        import torch
        use_gpu = torch.cuda.is_available()
        print(f"🔄 EasyOCR - GPU: {use_gpu}")
        _reader = easyocr.Reader(['en', 'ar'], gpu=use_gpu)
        print("✅ EasyOCR ready")
    return _reader


def read_id_from_frame(frame: np.ndarray, badge_bbox: list) -> str | None:
    x1, y1, x2, y2 = [int(v) for v in badge_bbox]

    pad = 25
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        print("OCR: empty crop")
        return None

    # تحسين بسيط قبل OCR
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    reader = _get_reader()
    results = reader.readtext(thresh)

    print("OCR raw results:", results)

    if not results:
        return None

    results.sort(key=lambda r: r[2], reverse=True)

    for _, text, conf in results:
        if conf < 0.25:
            continue
        cleaned = _clean_id(text)
        print("OCR candidate:", text, "->", cleaned, "conf=", conf)
        if cleaned:
            return cleaned

    return None


def _clean_id(text: str) -> str | None:
    text = text.upper().strip().replace(" ", "").replace("_", "-")

    # أمثلة:
    # EMP001 -> EMP-001
    # EMP-001 -> EMP-001
    # MD221 -> MD-221
    # MD-221 -> MD-221

    m = re.match(r'^([A-Z]+)-?(\d{2,})$', text)
    if m:
        prefix = m.group(1)
        number = m.group(2)
        return f"{prefix}-{number}"

    return None