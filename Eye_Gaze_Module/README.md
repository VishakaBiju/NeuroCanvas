# 👁️ Eye Gaze Emotion Recognition Module

This module predicts human emotional states using eye movement patterns and pupil behavior.

---

## 🔍 Overview

The system uses gaze coordinates and pupil dilation data to classify emotions into:

- Anger
- Disgust
- Fear
- Happy
- Sad

---

## ⚙️ Features Extracted

- avg_x → Average horizontal gaze position  
- avg_y → Average vertical gaze position  
- std_x → Horizontal variability  
- std_y → Vertical variability  
- avg_pupil → Average pupil diameter  
- saccade_speed → Eye movement speed  

---

## 🧠 Models Used

| Model            | Accuracy |
|------------------|---------|
| Random Forest    | 90%     |
| CatBoost         | 94.92%  |
| XGBoost          | 95.00%  |
| MLP              | 97.62%  |

---

## 📊 Results

### Random Forest
<img src="Outputs/rf_confusion.png" width="400"/>

### XGBoost
<img src="Outputs/xgb_confusion.png" width="400"/>

### MLP
<img src="Outputs/mlp_confusion.png" width="400"/>

### CatBoost
<img src="Outputs/catboost_confusion.png" width="400"/>

---

## 🎯 Real-Time System (OCULUS)

- Integrated with **Tobii Eye Tracker**
- Live gaze tracking + prediction
- Heatmap + scanpath visualization
- Flask backend + HTML dashboard

---

## 🎥 Demo

<p align="center">
  <video src="Demo/realtime_demo.mp4" width="600" controls></video>
</p>

---

## 🚀 How to Run

```bash
pip install opencv-python flask joblib tobii_research python-vlc
python app_server_tobii.py
