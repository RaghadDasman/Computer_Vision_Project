"""
tracker/tracker.py
------------------
DeepSORT wrapper - يعطي كل عامل track_id ثابت
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class WorkerTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,   # كم فريم نحتفظ بالـ track بدون كشف
            n_init=1,     # كم مرة نشوف الشخص قبل نثبّت الـ ID
        )

    def update(self, detections, frame):
        """
        detections: list of YOLO boxes للأشخاص فقط
        returns: list of dicts {track_id, bbox}
        """
        if not detections:
            return []

        # حوّل للصيغة اللي يطلبها DeepSORT: ([x1,y1,x2,y2], conf, class)
        raw = []
        for det in detections:
            bbox = det.xyxy[0].cpu().numpy().tolist()   # [x1,y1,x2,y2]
            conf = float(det.conf[0].cpu().numpy())
            raw.append((bbox, conf, 0))                 # class=0 person

        tracks = self.tracker.update_tracks(raw, frame=frame)

        result = []
        for t in tracks:
            if True:
                result.append({
                    "track_id": t.track_id,
                    "bbox":     t.to_ltrb().tolist()   # [x1,y1,x2,y2]
                })
        return result


# singleton
worker_tracker = WorkerTracker()
