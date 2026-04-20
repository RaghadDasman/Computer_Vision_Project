"""
logic/pipeline.py
نسخة محسنة:
- person detection مستقل
- tracker للفيديو فقط
- image mode بدون tracker
- رسم كل detections
- ربط البطاقة بالعامل بشكل أذكى
"""

from ultralytics import YOLO
import numpy as np
import cv2
import torch

from logic.ppe_checker import check_ppe
from logic.ladder_checker import full_ladder_check
from logic.id_reader import read_id_from_frame
from tracker.tracker import worker_tracker
from database.db import save_log, get_employee_info


PPE_CLASSES = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
    5: "goggles",
    6: "no_goggles",
    7: "gloves",
    8: "no_gloves",
}

ID_CLASSES = {0: "worker_id_tag"}
LADDER_CLASSES = {0: "ladder"}

COLOR_SAFE = (0, 200, 0)
COLOR_DANGER = (0, 0, 220)
COLOR_UNKNOWN = (180, 180, 0)
COLOR_PPE = (255, 255, 0)
COLOR_LADDER = (255, 165, 0)
COLOR_PERSON = (255, 0, 255)
COLOR_ID = (255, 255, 0)


class SafetyPipeline:
    def __init__(
        self,
        ppe_weights: str = "weights/ppe_best.pt",
        id_weights: str = "weights/id_best.pt",
        ladder_weights: str = "weights/ladder_best.pt",
        pose_weights: str = "yolov8s-pose.pt",
        person_weights: str = "yolov8s.pt",
        conf_threshold: float = 0.20,
        use_tracker_for_video: bool = True,
    ):
        torch.load = lambda *args, **kwargs: torch.serialization.load(
            *args, **kwargs, weights_only=False
        )

        print("🔄 Loading models...")
        self.ppe_model = YOLO(ppe_weights)
        self.id_model = YOLO(id_weights)
        self.ladder_model = YOLO(ladder_weights)
        self.pose_model = YOLO(pose_weights)
        self.person_model = YOLO(person_weights)

        self.conf = conf_threshold
        self.use_tracker_for_video = use_tracker_for_video
        self.worker_id_map: dict[int, str] = {}
        self.prev_frame = None
        print("✅ All models loaded")

    def process_frame(self, frame: np.ndarray):
        ppe_results = self.ppe_model(frame, conf=self.conf, verbose=False)[0]
        id_results = self.id_model(frame, conf=0.08, verbose=False)[0]        
        ladder_results = self.ladder_model(frame, conf=self.conf, verbose=False)[0]
        person_results = self.person_model(frame, conf=0.30, verbose=False)[0]

        ppe_detections = ppe_results.boxes
        id_detections = id_results.boxes
        print("ID detections count:", len(id_detections))
        for det in id_detections:
            print("ID cls:", int(det.cls[0]), "conf:", float(det.conf[0].cpu().numpy()))
        ladder_boxes = self._get_class_boxes(ladder_results.boxes, 0)
        person_boxes = self._get_person_boxes_from_general_detector(person_results.boxes)

        is_video_mode = self._looks_like_video_stream(frame)

        if is_video_mode and self.use_tracker_for_video:
            tracks = worker_tracker.update(person_boxes, frame)
        else:
            tracks = self._boxes_to_fake_tracks(person_boxes)

        pose_keypoints = None
        if ladder_boxes:
            pose_out = self.pose_model(frame, conf=self.conf, verbose=False)[0]
            if pose_out.keypoints is not None and len(pose_out.keypoints.data) > 0:
                pose_keypoints = pose_out.keypoints.data.cpu().numpy()

        workers_output = []
        annotated = frame.copy()

        annotated = self._draw_person_boxes(annotated, tracks)
        annotated = self._draw_ppe_detections(annotated, ppe_detections)
        annotated = self._draw_id_detections(annotated, id_detections)
        annotated = self._draw_ladders(annotated, ladder_boxes)

        for track in tracks:
            tid = track["track_id"]
            pbox = track["bbox"]

            result = self._process_worker(
                frame=frame,
                track_id=tid,
                person_bbox=pbox,
                ppe_dets=ppe_detections,
                id_dets=id_detections,
                ladder_boxes=ladder_boxes,
                all_kps=pose_keypoints,
            )

            workers_output.append(result)

            if result["alerts"]:
                self._save_to_db(result)

            annotated = self._draw_worker(annotated, result)

        result_dict = {
            "workers": workers_output,
            "total_workers": len(workers_output),
            "has_ladder": len(ladder_boxes) > 0,
            "mode": "video" if is_video_mode else "image",
        }

        self.prev_frame = frame.copy()
        return annotated, result_dict

    def _draw_person_boxes(self, frame, tracks):
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 1)
            cv2.putText(
                frame,
                f"person-{track['track_id']}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_PERSON,
                1,
            )
        return frame

    def _draw_ppe_detections(self, frame, ppe_detections):
        for det in ppe_detections:
            cls_id = int(det.cls[0])
            label = PPE_CLASSES.get(cls_id, "unknown")

            if label == "person":
                continue

            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].cpu().numpy().tolist()]
            conf = float(det.conf[0].cpu().numpy())

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PPE, 1)
            cv2.putText(
                frame,
                f"{label}: {conf:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_PPE,
                1,
            )
        return frame

    def _draw_id_detections(self, frame, id_detections):
        for det in id_detections:
            if int(det.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = [int(v) for v in det.xyxy[0].cpu().numpy().tolist()]
            conf = float(det.conf[0].cpu().numpy())

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ID, 1)
            cv2.putText(
                frame,
                f"id_tag: {conf:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_ID,
                1,
            )
        return frame

    def _draw_ladders(self, frame, ladder_boxes):
        for lb in ladder_boxes:
            x1, y1, x2, y2 = [int(v) for v in lb]
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_LADDER, 2)
            cv2.putText(
                frame,
                "Ladder",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_LADDER,
                1,
            )
        return frame

    def _draw_worker(self, frame, result):
        x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
        color = COLOR_SAFE if result["is_safe"] else COLOR_DANGER

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{result['employee_id']}"
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if result["ppe"]["missing"]:
            missing_txt = "Missing: " + ", ".join(result["ppe"]["missing"])
            cv2.putText(
                frame,
                missing_txt,
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_DANGER,
                1,
            )

        if result["ladder"] and result["ladder"]["has_alert"]:
            cv2.putText(
                frame,
                "! Ladder Alert",
                (x1, y2 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 140, 255),
                1,
            )

        return frame

    def _process_worker(
        self,
        frame,
        track_id,
        person_bbox,
        ppe_dets,
        id_dets,
        ladder_boxes,
        all_kps,
    ):
        if track_id not in self.worker_id_map:
            emp_id = self._read_worker_id(frame, id_dets, person_bbox)
            if emp_id:
                self.worker_id_map[track_id] = emp_id
                print(f"🪪 {emp_id} (track={track_id})")

        emp_id = self.worker_id_map.get(track_id, f"UNKNOWN-{track_id}")
        emp_info = get_employee_info(emp_id)
        ppe = check_ppe(ppe_dets, person_bbox)

        ladder_data = None
        if ladder_boxes:
            nearest = self._nearest_ladder(person_bbox, ladder_boxes)
            if nearest:
                kps = self._get_person_keypoints(person_bbox, all_kps)
                ladder_data = full_ladder_check(
                    frame=frame,
                    person_bbox=person_bbox,
                    ladder_bbox=nearest,
                    keypoints=kps,
                )

        alerts = []
        if not ppe["compliant"]:
            alerts.append(
                {
                    "type": "PPE_VIOLATION",
                    "msg": f"العامل {emp_id} لا يرتدي: {'، '.join(ppe['missing_ar'])}",
                    "missing": ppe["missing"],
                }
            )

        if ladder_data and ladder_data["has_alert"]:
            for msg in ladder_data["alerts"]:
                alerts.append(
                    {
                        "type": "LADDER_VIOLATION",
                        "msg": f"{emp_id} - {msg}",
                    }
                )

        return {
            "track_id": track_id,
            "employee_id": emp_id,
            "employee_name": emp_info.get("name", "-"),
            "bbox": person_bbox,
            "ppe": ppe,
            "ladder": ladder_data,
            "alerts": alerts,
            "is_safe": len(alerts) == 0,
        }

    def _read_worker_id(self, frame, id_dets, person_bbox):
        badge_boxes = []

        for det in id_dets:
            if int(det.cls[0]) != 0:
                continue
            badge_box = det.xyxy[0].cpu().numpy().tolist()
            badge_boxes.append(badge_box)

        if not badge_boxes:
            print("No ID detections found")
            return None

        # إذا الصورة فيها badge واحدة فقط خذيها مباشرة
        if len(badge_boxes) == 1:
            emp_id = read_id_from_frame(frame, badge_boxes[0])
            print("Single badge found, OCR result:", emp_id)
            return emp_id.strip().upper() if emp_id else None

        # إذا فيه أكثر من badge اختاري الأقرب للعامل
        px = (person_bbox[0] + person_bbox[2]) / 2
        py = (person_bbox[1] + person_bbox[3]) / 2

        best_box = None
        best_dist = float("inf")

        for badge_box in badge_boxes:
            bx = (badge_box[0] + badge_box[2]) / 2
            by = (badge_box[1] + badge_box[3]) / 2
            dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

            print("ID distance:", dist, "badge_box:", badge_box, "person_bbox:", person_bbox)

            if dist < best_dist:
                best_dist = dist
                best_box = badge_box

        if best_box is None:
            print("No badge matched this worker")
            return None

        emp_id = read_id_from_frame(frame, best_box)
        print("Closest badge OCR result:", emp_id)

        return emp_id.strip().upper() if emp_id else None

    def _get_person_boxes_from_general_detector(self, boxes):
        person_boxes = []
        for b in boxes:
            cls_id = int(b.cls[0])
            if cls_id == 0:
                person_boxes.append(b)
        return person_boxes

    def _boxes_to_fake_tracks(self, person_boxes):
        tracks = []
        for i, box in enumerate(person_boxes):
            bbox = box.xyxy[0].cpu().numpy().tolist()
            tracks.append(
                {
                    "track_id": i,
                    "bbox": bbox,
                }
            )
        return tracks

    def _looks_like_video_stream(self, frame):
        if self.prev_frame is None:
            return False

        if self.prev_frame.shape != frame.shape:
            return False

        diff = cv2.absdiff(self.prev_frame, frame)
        mean_diff = float(np.mean(diff))
        return mean_diff > 1.5

    def _get_class_boxes(self, boxes, cls_id):
        return [
            b.xyxy[0].cpu().numpy().tolist()
            for b in boxes
            if int(b.cls[0]) == cls_id
        ]

    def _nearest_ladder(self, person_bbox, ladder_boxes):
        px = (person_bbox[0] + person_bbox[2]) / 2
        py = (person_bbox[1] + person_bbox[3]) / 2

        nearest = None
        min_dist = float("inf")

        for lb in ladder_boxes:
            lx = (lb[0] + lb[2]) / 2
            ly = (lb[1] + lb[3]) / 2
            d = ((px - lx) ** 2 + (py - ly) ** 2) ** 0.5

            if d < min_dist and d < 300:
                min_dist = d
                nearest = lb

        return nearest

    def _get_person_keypoints(self, person_bbox, all_kps):
        if all_kps is None or len(all_kps) == 0:
            return None

        best_kps = None
        best_iou = 0.0

        for kps in all_kps:
            valid = kps[kps[:, 2] > 0.3]
            if len(valid) == 0:
                continue

            kp_box = [
                valid[:, 0].min(),
                valid[:, 1].min(),
                valid[:, 0].max(),
                valid[:, 1].max(),
            ]

            iou = self._iou(person_bbox, kp_box)
            if iou > best_iou:
                best_iou = iou
                best_kps = kps

        return best_kps if best_iou > 0.2 else None

    @staticmethod
    def _boxes_overlap(b1, b2, margin=40):
        return (
            b1[0] - margin < b2[2]
            and b1[2] + margin > b2[0]
            and b1[1] - margin < b2[3]
            and b1[3] + margin > b2[1]
        )

    @staticmethod
    def _iou(A, B):
        xA, yA = max(A[0], B[0]), max(A[1], B[1])
        xB, yB = min(A[2], B[2]), min(A[3], B[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0

        areaA = (A[2] - A[0]) * (A[3] - A[1])
        areaB = (B[2] - B[0]) * (B[3] - B[1])

        return inter / (areaA + areaB - inter)

    def _save_to_db(self, r):
        ppe = r["ppe"]
        lad = r.get("ladder") or {}
        tp = lad.get("three_point") or {}

        save_log(
            {
                "employee_id": r["employee_id"],
                "helmet": "helmet" in ppe["present"],
                "vest": "vest" in ppe["present"],
                "gloves": "gloves" in ppe["present"],
                "goggles": "goggles" in ppe["present"],
                "boots": False,
                "ppe_compliant": ppe["compliant"],
                "near_ladder": lad.get("zone") is not None,
                "ladder_angle": lad.get("angle"),
                "ladder_zone": lad.get("zone"),
                "three_point_ok": tp.get("safe"),
                "contact_count": tp.get("label"),
                "alert_sent": True,
                "alert_msg": " | ".join(a["msg"] for a in r["alerts"]),
            }
        )


pipeline = SafetyPipeline()