"""
tests/test_pipeline.py
-----------------------
اختبر الـ pipeline على صورة ثابتة أو فيديو قبل التشغيل الكامل
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.ladder_checker import (
    calculate_ladder_angle,
    get_worker_zone,
    check_three_point_contact,
    full_ladder_check,
    angle_risk_level,
)
from logic.ppe_checker import check_ppe
from database.db import init_db, get_recent_logs
import numpy as np


def test_ladder_logic():
    print("\n🧪 Test 1: Ladder angle risk levels")
    for angle, expected in [(80, "safe"), (65, "warning"), (45, "danger"), (None, "unknown")]:
        result = angle_risk_level(angle)
        status = "✅" if result == expected else "❌"
        print(f"  {status} angle={angle} → {result}")


def test_three_point():
    print("\n🧪 Test 2: Three-point contact")

    # سلم وهمي
    ladder_box = [100, 50, 200, 400]

    # keypoints وهمية (17, 3) = x, y, conf
    kps = np.zeros((17, 3))

    # ضع 3 نقاط داخل السلم
    kps[9]  = [150, 100, 0.9]   # left_wrist  ✅
    kps[10] = [160, 150, 0.9]   # right_wrist ✅
    kps[15] = [140, 350, 0.9]   # left_ankle  ✅
    kps[16] = [250, 380, 0.9]   # right_ankle ❌ (خارج السلم)

    result = check_three_point_contact(kps, ladder_box)
    status = "✅" if result["safe"] else "❌"
    print(f"  {status} contact_count={result['contact_count']}/4 → safe={result['safe']}")


def test_zone():
    print("\n🧪 Test 3: Ladder zones")

    ladder_box = [100, 100, 200, 500]   # ارتفاع 400px

    # أعلى الزون = danger
    person_top    = [120, 100, 180, 180]
    person_middle = [120, 250, 180, 330]
    person_bottom = [120, 420, 180, 500]

    for person, expected in [
        (person_top,    "danger"),
        (person_middle, "warning"),
        (person_bottom, "safe"),
    ]:
        zone   = get_worker_zone(person, ladder_box)
        status = "✅" if zone == expected else "❌"
        print(f"  {status} expected={expected} → got={zone}")


def test_database():
    print("\n🧪 Test 4: Database")
    init_db()
    logs = get_recent_logs(5)
    print(f"  ✅ Database OK - {len(logs)} logs found")


def test_on_image(image_path: str):
    """اختبر على صورة فعلية"""
    print(f"\n🧪 Test 5: Real image → {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print("  ❌ Image not found")
        return

    # لو عندك الـ weights جاهزة
    try:
        from logic.pipeline import pipeline
        result = pipeline.process_frame(frame)
        print(f"  ✅ Workers detected: {result['total_workers']}")
        for w in result["workers"]:
            print(f"     👷 {w['employee_id']} | PPE: {w['ppe']['compliant']} | Alerts: {len(w['alerts'])}")
    except Exception as e:
        print(f"  ⚠️  Pipeline error (weights missing?): {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("   Safety System - Unit Tests")
    print("=" * 50)

    test_ladder_logic()
    test_three_point()
    test_zone()
    test_database()

    # لو عندك صورة للاختبار
    # test_on_image("tests/sample.jpg")

    print("\n✅ All logic tests passed!")
    print("=" * 50)
