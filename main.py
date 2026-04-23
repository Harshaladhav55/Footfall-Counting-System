import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os

# ========================= CONFIGURATION =========================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.25
DEVICE = "cpu"
VIDEO_FILE_PATH = "Sanjivani.mp4"      # Change this when you add new video

# Choose input source
print("\n" + "="*50)
print("🚀 FootFall Counting System + Improved Step Counter")
print("="*50)
print("Choose input source:")
print("1. Webcam (Live Camera)")
print(f"2. Video File ({VIDEO_FILE_PATH})")
choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    VIDEO_SOURCE = 0
    print("✅ Using Webcam")
elif choice == "2":
    VIDEO_SOURCE = VIDEO_FILE_PATH
    if not os.path.exists(VIDEO_SOURCE):
        print(f"❌ ERROR: File '{VIDEO_SOURCE}' not found!")
        exit()
    print(f"✅ Using Video File: {VIDEO_SOURCE}")
else:
    VIDEO_SOURCE = 0

# Create model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)
assert cap.isOpened(), "Cannot open video!"

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

COUNTING_LINE_X = w // 2

# Video writer
out = cv2.VideoWriter('output/footfall_output.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Tracking variables
states = {}
in_count = 0
out_count = 0
track_history = defaultdict(list)
step_counts = defaultdict(int)
last_centroid_y = {}
cooldown = defaultdict(int)          # NEW: prevents counting too fast

STEP_THRESHOLD = 4                   # Lowered for better detection
COOLDOWN_FRAMES = 5                  # NEW: one step every 5 frames

print("🚀 System Started with Improved Step Counter")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          classes=[0], conf=CONF_THRESHOLD, 
                          device=DEVICE, verbose=False)

    cv2.line(frame, (COUNTING_LINE_X, 0), (COUNTING_LINE_X, h), (0, 255, 255), 3)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # IN/OUT Counting
            current_state = 'left' if centroid_x < COUNTING_LINE_X else 'right'
            if track_id not in states:
                states[track_id] = current_state
            else:
                if states[track_id] != current_state:
                    if current_state == 'right':
                        in_count += 1
                    else:
                        out_count += 1
                    states[track_id] = current_state

            # IMPROVED STEP COUNTING
            cooldown[track_id] = max(0, cooldown[track_id] - 1)
            if track_id in last_centroid_y and cooldown[track_id] == 0:
                delta_y = abs(centroid_y - last_centroid_y[track_id])
                if delta_y > STEP_THRESHOLD:
                    step_counts[track_id] += 1
                    cooldown[track_id] = COOLDOWN_FRAMES   # cooldown

            last_centroid_y[track_id] = centroid_y

            # Trajectory
            track_history[track_id].append((centroid_x, centroid_y))
            if len(track_history[track_id]) > 20:
                track_history[track_id].pop(0)

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Steps:{step_counts[track_id]}", (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)

            for i in range(1, len(track_history[track_id])):
                pt1 = track_history[track_id][i-1]
                pt2 = track_history[track_id][i]
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)

    # Live Statistics
    cv2.putText(frame, f"IN: {in_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"TOTAL: {in_count + out_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(frame, "CPU Mode - YOLOv8 + ByteTrack + Step Counter", (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("FootFall Counter", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Final Report
print("\n" + "="*60)
print("✅ FINISHED - STEP COUNT REPORT")
print("="*60)
for tid, steps in step_counts.items():
    print(f"Person ID {tid:3d}  →  {steps} steps")
print(f"\nIN: {in_count} | OUT: {out_count} | Total Footfall: {in_count + out_count}")
print("="*60)