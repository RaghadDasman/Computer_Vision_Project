---
title: project-Cv
emoji: "🚀"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI-Based Industrial Safety Monitoring System Using Computer Vision

An intelligent real-time worker safety monitoring system for industrial environments, built on YOLOv8 and FastAPI.

🔗 **[Live Demo](https://mansouralhenaki-project-cv.hf.space/app/login.html)**

---

##  Features

- Personal Protective Equipment (PPE) detection — helmet, vest, goggles, gloves
- Worker identity recognition via ID badges
- Ladder detection with body pose analysis to prevent fall accidents
- Real-time worker tracking using Deep SORT
- Interactive dashboard with live stats and violation alerts
- Supports live camera feed and pre-recorded video

##  Models

| Model | Purpose |
|-------|---------|
| YOLOv8 (ppe_best.pt) | PPE detection |
| YOLOv8 (id_best.pt) | ID badge detection |
| YOLOv8 (ladder_best.pt) | Ladder detection |
| YOLOv8s-pose | Body pose estimation |
| YOLOv8s | General person detection |

##  Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Computer Vision:** Ultralytics YOLOv8, OpenCV, EasyOCR
- **Tracking:** Deep SORT
- **Database:** SQLite + SQLAlchemy
- **Frontend:** HTML / CSS / JavaScript
- **Deployment:** Docker on Hugging Face Spaces
