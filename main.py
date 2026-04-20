"""
main.py - النسخة النهائية مع دعم الفيديو والصور
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import asyncio
import numpy as np
import threading
import tempfile
import os
import base64
from pathlib import Path

from database.db import init_db, get_recent_logs, get_employee_info, register_employee, verify_login

# ── مجلد الفيديوهات الجاهزة ──
VIDEOS_DIR = Path("demo_videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# ── مجلد الوسائط (صور أو فيديو يُحطّ مباشرة) ──
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ── مدير مصدر الفيديو (كاميرا أو فيديو أو صورة) ──
class VideoSource:
    def __init__(self):
        self.cap        = None
        self.lock       = threading.Lock()
        self.frame      = None
        self.running    = False
        self.source     = None
        self._img_frame = None  # لو المصدر صورة ثابتة

    def start(self, source=0):
        self.stop()
        self.source     = source
        self._img_frame = None

        # لو صورة — حمّلها مباشرة
        if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(source)
            if img is not None:
                self._img_frame = img
                with self.lock:
                    self.frame = img
                print(f"🖼️  Image source: {Path(source).name}")
                self.running = True
                return

        self.cap     = cv2.VideoCapture(source)
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print(f"📷 Video source: {source}")

    def _loop(self):
        while self.running:
            if self._img_frame is not None:
                # صورة ثابتة — ما نحتاج نقرأ شيء
                import time; time.sleep(0.03)
                continue
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
                    continue
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def switch(self, source):
        self.start(source)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None


video_source = VideoSource()

# صور مُحلَّلة لكل كاميرا (slot 0-3)
cam_frames: dict[int, bytes] = {}
BLACK_FRAME: bytes = b''


def _make_black_jpeg(w=640, h=480) -> bytes:
    black = np.zeros((h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', black)
    return buf.tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global BLACK_FRAME
    init_db()
    from logic.pipeline import pipeline
    app.state.pipeline = pipeline
    BLACK_FRAME = _make_black_jpeg()

    # تحقق من media/ — صور أو فيديو
    images = sorted([f for f in MEDIA_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    videos = sorted([f for f in MEDIA_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTS])

    if images:
        # حلّل كل صورة واحفظ النتيجة لكل slot
        for i, img_path in enumerate(images[:4]):
            frame = cv2.imread(str(img_path))
            if frame is not None:
                annotated, _ = pipeline.process_frame(frame)
                _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cam_frames[i] = buf.tobytes()
                print(f"🖼️  Slot {i}: {img_path.name}")
    elif videos:
        video_source.start(str(videos[0]))
        print(f"📂 Media video: {videos[0].name}")
    else:
        demo = list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.avi"))
        if demo:
            video_source.start(str(demo[0]))
            print(f"🎬 Demo video: {demo[0].name}")
        else:
            video_source.start(0)

    print("🚀 Safety System started")
    yield
    video_source.stop()
    print("🛑 Stopped")


app = FastAPI(title="Factory Safety System", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════
# 1. LIVE STREAM
# ══════════════════════════════════════════
def generate_frames(pipeline):
    while True:
        frame = video_source.get_frame()
        if frame is None:
            continue
        annotated, _ = pipeline.process_frame(frame)
        ret, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


def _stream_static(jpeg: bytes):
    while True:
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(app.state.pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed/{slot}")
def video_feed_slot(slot: int):
    """كاميرا محددة — صورة محللة أو شاشة سوداء أو البث المباشر"""
    if cam_frames:
        jpeg = cam_frames.get(slot, BLACK_FRAME)
        return StreamingResponse(
            _stream_static(jpeg),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    # وضع الفيديو/الكاميرا — كل الـ slots نفس البث
    return StreamingResponse(
        generate_frames(app.state.pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/stream")
async def data_stream(ws: WebSocket):
    await ws.accept()
    pipeline = ws.app.state.pipeline

    try:
        while True:
            result = None

            # إذا عندنا صور static في slots
            if cam_frames:
                # استخدم أول صورة من media لتحريك الإحصائيات
                images = sorted([f for f in MEDIA_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTS])
                if images:
                    frame = cv2.imread(str(images[0]))
                    if frame is not None:
                        _, result = pipeline.process_frame(frame)

            else:
                # وضع الفيديو/الكاميرا
                frame = video_source.get_frame()
                if frame is not None:
                    _, result = pipeline.process_frame(frame)

            if result is not None:
                await ws.send_json(result)

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass

# ══════════════════════════════════════════
# 2. SWITCH SOURCE
# ══════════════════════════════════════════
@app.post("/source/camera")
def use_camera():
    video_source.switch(0)
    return {"source": "camera", "status": "switched"}


@app.post("/source/video/{filename}")
def use_video(filename: str):
    path = VIDEOS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"الفيديو '{filename}' غير موجود")
    video_source.switch(str(path))
    return {"source": filename, "status": "switched"}


@app.get("/source/videos")
def list_videos():
    videos = (
        [f.name for f in VIDEOS_DIR.glob("*.mp4")] +
        [f.name for f in VIDEOS_DIR.glob("*.avi")] +
        [f.name for f in VIDEOS_DIR.glob("*.mov")]
    )
    return {"videos": videos, "current": str(video_source.source)}


@app.post("/source/upload_video")
async def upload_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(400, "نوع الملف غير مدعوم. استخدم mp4/avi/mov")
    save_path = VIDEOS_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(await file.read())
    video_source.switch(str(save_path))
    return {"status": "uploaded_and_playing", "filename": file.filename}


# ══════════════════════════════════════════
# 3. ANALYZE IMAGE
# ══════════════════════════════════════════
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "الملف ليس صورة صالحة")

    pipeline          = app.state.pipeline
    annotated, result = pipeline.process_frame(frame)

    _, buf  = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "result":          result,
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "total_workers":   result["total_workers"],
        "has_ladder":      result["has_ladder"],
        "safe_workers":    sum(1 for w in result["workers"] if w["is_safe"]),
        "unsafe_workers":  sum(1 for w in result["workers"] if not w["is_safe"]),
    }


# ══════════════════════════════════════════
# 4. ANALYZE VIDEO
# ══════════════════════════════════════════
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), sample_rate: int = 30):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".mp4", ".avi", ".mov"]:
        raise HTTPException(400, "استخدم mp4/avi/mov")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pipeline     = app.state.pipeline
        cap          = cv2.VideoCapture(tmp_path)
        total_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        total_f      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        violations   = []
        workers_seen = set()
        frame_idx    = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue
            _, result = pipeline.process_frame(frame)
            timestamp = round(frame_idx / total_fps, 1)
            for w in result["workers"]:
                workers_seen.add(w["employee_id"])
                if w["alerts"]:
                    violations.append({
                        "time_sec":    timestamp,
                        "employee_id": w["employee_id"],
                        "alerts":      [a["msg"] for a in w["alerts"]],
                    })
        cap.release()

        return {
            "video":            file.filename,
            "duration_sec":     round(total_f / total_fps, 1),
            "frames_analyzed":  frame_idx // sample_rate,
            "workers_seen":     list(workers_seen),
            "total_violations": len(violations),
            "violations":       violations,
        }
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════
# 5. AUTH
# ══════════════════════════════════════════
class RegisterRequest(BaseModel):
    employee_id: str
    name: str
    department: str
    password: str


class LoginRequest(BaseModel):
    employee_id: str
    password: str


@app.post("/register")
def register(req: RegisterRequest):
    result = register_employee(req.employee_id, req.name, req.department, req.password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/login")
def login(req: LoginRequest):
    result = verify_login(req.employee_id, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return result


# ══════════════════════════════════════════
# 6. DATABASE & STATUS
# ══════════════════════════════════════════
@app.get("/")
def root():
    videos = (
        [f.name for f in VIDEOS_DIR.glob("*.mp4")] +
        [f.name for f in VIDEOS_DIR.glob("*.avi")]
    )
    return {
        "status":  "running ✅",
        "source":  str(video_source.source),
        "videos":  videos,
        "endpoints": {
            "live_feed":      "GET  /video_feed",
            "switch_camera":  "POST /source/camera",
            "switch_video":   "POST /source/video/{filename}",
            "upload_video":   "POST /source/upload_video",
            "analyze_image":  "POST /analyze/image",
            "analyze_video":  "POST /analyze/video",
            "logs":           "GET  /logs",
            "status":         "GET  /status",
            "docs":           "GET  /docs",
        }
    }


@app.get("/status")
def get_status():
    pipeline = app.state.pipeline
    return {
        "active_workers": len(pipeline.worker_id_map),
        "workers": [{"track_id": tid, "employee_id": eid}
                    for tid, eid in pipeline.worker_id_map.items()],
        "source": str(video_source.source),
    }


@app.get("/logs")
def get_logs(limit: int = 50):
    return {"logs": get_recent_logs(limit)}


@app.get("/logs/{employee_id}")
def get_employee_logs(employee_id: str, limit: int = 20):
    from database.db import SessionLocal, SafetyLog
    db   = SessionLocal()
    logs = (db.query(SafetyLog)
            .filter(SafetyLog.employee_id == employee_id.upper())
            .order_by(SafetyLog.timestamp.desc())
            .limit(limit).all())
    db.close()
    if not logs:
        raise HTTPException(404, "لا توجد سجلات")
    return {
        "employee": get_employee_info(employee_id.upper()),
        "logs": [{"timestamp": str(l.timestamp), "alert_msg": l.alert_msg,
                  "ppe_compliant": l.ppe_compliant} for l in logs],
    }


@app.get("/employees")
def get_employees():
    from database.db import SessionLocal, Employee
    db   = SessionLocal()
    emps = db.query(Employee).filter_by(active=True).all()
    db.close()
    return {"employees": [{"id": e.employee_id, "name": e.name,
                           "department": e.department} for e in emps]}


# ── Static Frontend ──────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")


#http://127.0.0.1:8000/app
#python -m uvicorn main:app --reload