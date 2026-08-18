# Python Data & Processing Pipeline

## 1. Overview

The Footfall Analytics System uses a Python-based computer vision
pipeline to process video data and identify people, track their
movement, and calculate entry and exit counts.

The complete processing flow is:

Input Video / Webcam
↓
OpenCV Video Capture
↓
Frame Extraction
↓
YOLOv8 Person Detection
↓
ByteTrack Object Tracking
↓
Bounding Box / Centroid Processing
↓
Line Crossing Analysis
↓
Entry / Exit Counting
↓
Processed Video Output

---

## 2. Input Layer

The system accepts video data from:

- Video files
- Webcam

OpenCV is used to open the selected input source and read video
frames sequentially.

---

## 3. Frame Processing

The input video is processed frame-by-frame.

For every frame:

1. The frame is captured using OpenCV.
2. The frame is passed to the object detection model.
3. Detected people are identified.
4. Tracking information is generated.
5. The movement of tracked people is analyzed.
6. Entry and exit counts are updated when applicable.
7. The processed frame can be written to an output video.

---

## 4. Person Detection

YOLOv8 is used as the object detection model.

The system focuses on the person class because the objective is to
count and track people.

The detector produces bounding boxes around detected people.

Each detection contains information such as:

- Bounding box coordinates
- Detection confidence
- Object class

---

## 5. Object Tracking

ByteTrack is used to track detected people across consecutive video
frames.

Tracking allows the system to associate detections from different
frames with a tracking ID.

For example:

Frame 1 → Person ID 1
Frame 2 → Person ID 1
Frame 3 → Person ID 1

This prevents the system from treating the same person as a completely
new person in every frame.

---

## 6. Centroid Calculation

For each tracked person's bounding box, the system calculates a
central point called the centroid.

The centroid is used to understand the person's movement between
frames.

Conceptually:

Bounding Box
↓
Center Point
↓
Previous Position
↓
Current Position
↓
Movement Direction

---

## 7. Entry and Exit Detection

A virtual counting line is defined in the video.

The system compares the position of a tracked person's centroid with
the counting line.

When a tracked person crosses the line, the system determines the
direction of movement.

Example:

Left → Right
= Entry

Right → Left
= Exit

The corresponding count is then updated.

---

## 8. Trajectory Tracking

The system maintains movement history for tracked people.

The trajectory represents the path followed by a person during the
video.

This information can be used to visualize movement and understand
basic visitor flow.

---

## 9. Step / Movement Analysis

The system also contains movement-based step counting logic.

Movement is estimated using changes in the tracked person's centroid
position.

A cooldown mechanism is used to reduce repeated counting caused by
small movements between consecutive frames.

---

## 10. Output Layer

The processed frames can be written to an output video.

The output can contain visual information such as:

- Person bounding boxes
- Tracking IDs
- Counting line
- Entry count
- Exit count
- Movement information
- Trajectory information

---

## 11. Complete Pipeline

The complete system can be represented as:

Video / Webcam
↓
OpenCV
↓
Frame Extraction
↓
YOLOv8
↓
Person Detection
↓
ByteTrack
↓
Tracking IDs
↓
Centroid Calculation
↓
Movement Analysis
↓
Line Crossing Detection
↓
IN / OUT Counting
↓
Processed Video

---

## 12. Processing Considerations

The processing pipeline depends on the quality of the input video.

Factors such as:

- Lighting
- Camera angle
- Resolution
- Crowd density
- Occlusion
- Motion blur

can affect detection and tracking performance.

The system therefore requires suitable video input for reliable
footfall counting.