# Intelligence Layer — YOLOv8 + ByteTrack

## 1. Overview

The intelligence layer is responsible for understanding the visual
information contained in the input video.

The system uses a combination of:

- YOLOv8 for person detection
- ByteTrack for multi-object tracking
- Centroid-based movement analysis
- Line-crossing logic for entry and exit counting

The intelligence pipeline is:

Video Frame
↓
YOLOv8
↓
Person Detection
↓
ByteTrack
↓
Tracking ID
↓
Centroid Calculation
↓
Movement Direction
↓
Line Crossing
↓
IN / OUT Count

---

## 2. Person Detection — YOLOv8

YOLOv8 is used to detect people in each video frame.

The model identifies objects and provides:

- Bounding box coordinates
- Object class
- Detection confidence

For the footfall application, the system focuses on the person class.

The detection result becomes the input for the tracking stage.

---

## 3. Multi-Object Tracking — ByteTrack

ByteTrack is used to track people across consecutive video frames.

The tracking system assigns an ID to detected people.

For example:

Frame 1 → Person ID 1
Frame 2 → Person ID 1
Frame 3 → Person ID 1

This allows the system to understand that the detections belong to
the same person.

Tracking is important because counting every detection independently
would result in the same person being counted repeatedly across
multiple frames.

---

## 4. Movement Analysis

After a person is detected and assigned a tracking ID, the system
calculates the centroid of the person's bounding box.

The centroid provides an approximate position of the person.

The system compares the person's position across frames to determine
movement.

Previous Position
↓
Current Position
↓
Movement Direction

---

## 5. Footfall Counting Logic

A virtual counting line is defined in the monitored area.

When the centroid of a tracked person crosses the line, the system
checks the direction of movement.

Example:

Left → Right
= Entry

Right → Left
= Exit

The corresponding counter is then updated.

This converts visual movement into structured footfall information.

---

## 6. Why YOLOv8 Was Selected

YOLOv8 was selected because the project requires object detection
from video frames and needs to process the video efficiently.

The model provides the person locations required by the subsequent
tracking and counting stages.

The project focuses on practical video processing rather than
training a new object-detection model from scratch.

---

## 7. Why ByteTrack Was Selected

ByteTrack was selected because the project requires multi-object
tracking.

The system needs to maintain tracking identities across consecutive
frames so that the same person is not counted repeatedly.

Tracking IDs also allow the system to calculate movement and determine
the direction in which a person crosses the counting line.

---

## 8. Why a Traditional ML Model Was Not Used

A traditional tabular classification or regression model is not the
primary approach because the raw input is visual video data.

The main problem is object detection and tracking rather than
predicting a numerical value from structured tabular features.

Therefore, a computer-vision approach is more suitable.

---

## 9. Why an LLM Is Not Used

A Large Language Model is not required for the core footfall-counting
task.

The primary input is video and the required operations are:

- Object detection
- Object tracking
- Movement analysis
- Entry and exit counting

These operations are better handled by computer-vision models and
tracking algorithms.

An LLM could potentially be added in a future analytics or natural
language reporting feature, but it is not necessary for the core
system.

---

## 10. Model Limitations

The intelligence layer has several limitations:

- Detection performance depends on video quality.
- Heavy occlusion can affect tracking.
- Crowded scenes may cause tracking IDs to change.
- Poor lighting can reduce detection quality.
- Motion blur can affect detection.
- Camera movement can reduce tracking reliability.
- Incorrect counting-line placement can produce incorrect counts.

---

## 11. Failure Scenarios

Possible failure scenarios include:

### Person Not Detected

A person may not be detected because of poor visibility,
occlusion, or unsuitable camera positioning.

### Tracking ID Lost

A tracking identity may be lost when a person becomes temporarily
occluded or leaves the detection area.

### Incorrect Entry / Exit

An incorrectly positioned counting line or unstable tracking can
cause an incorrect entry or exit event.

---

## 12. Future Intelligence Improvements

Future versions could explore:

- Improved detection models
- Custom training for specific environments
- Improved tracking configuration
- Crowd-density analysis
- Occupancy estimation
- Time-based footfall prediction
- Anomaly detection
- Historical footfall forecasting