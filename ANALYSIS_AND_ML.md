# Data Analysis & AI/ML Approach

## 1. Overview

The Footfall Analytics System processes video data to detect and track
people and calculate movement across a defined counting line.

The analysis focuses on understanding:

- Number of detected people
- Number of tracked people
- Entry and exit events
- Movement patterns
- Tracking stability
- Detection quality
- Factors affecting counting accuracy

---

## 2. Data Characteristics

The primary input is video data.

The video consists of a sequence of image frames that contain people
moving through a monitored area.

Important characteristics include:

- Number of frames
- Frame resolution
- Frame rate
- Number of visible people
- Crowd density
- Camera position
- Lighting conditions
- Amount of movement

---

## 3. Important Features

The computer-vision pipeline extracts information from detected people.

Important features include:

### Bounding Box

The bounding box represents the detected person's location in a
video frame.

### Detection Confidence

The confidence score indicates how strongly the detection model
believes that the detected object belongs to the selected class.

### Tracking ID

Each tracked person receives a tracking ID that allows the system to
follow the person across multiple frames.

### Centroid Position

The center point of the bounding box is used to estimate the person's
position and movement.

### Movement Direction

The change in centroid position between frames helps determine the
direction of movement.

### Line Crossing

Crossing a predefined virtual line is used to identify entry and
exit events.

---

## 4. AI/ML Approach

The project uses a computer-vision-based AI approach rather than a
traditional tabular machine-learning model.

The main intelligence components are:

YOLOv8
↓
Person Detection
↓
ByteTrack
↓
Multi-Object Tracking
↓
Movement Analysis
↓
Footfall Counting

---

## 5. YOLOv8

YOLOv8 is used for real-time object detection.

In this project, the system focuses on detecting people in the input
video.

The model provides bounding boxes and confidence information for
detected objects.

The detected person information is then passed to the tracking
component.

---

## 6. ByteTrack

ByteTrack is used for multi-object tracking.

The purpose of tracking is to maintain the identity of detected
people across consecutive frames.

Without tracking, the same person could be detected repeatedly in
different frames and incorrectly counted multiple times.

ByteTrack helps associate detections across frames using tracking IDs.

---

## 7. Why This AI/ML Approach Was Selected

The project requires understanding visual information from video.

A traditional classification or regression model alone would not
solve the main problem because the system needs to:

- Locate people in images
- Track people across frames
- Determine movement
- Detect line crossing events

Therefore, object detection combined with multi-object tracking is
more appropriate for this use case.

---

## 8. Analysis of Movement

The movement of a tracked person can be analyzed using changes in the
centroid position.

For example:

Previous Position
↓
Current Position
↓
Movement Direction
↓
Counting Line Crossing
↓
Entry / Exit Event

This allows the system to convert visual movement into structured
footfall information.

---

## 9. Expected Patterns

The system can identify useful patterns such as:

- Number of entries over time
- Number of exits over time
- Periods with higher visitor movement
- Difference between entry and exit counts
- Movement through a monitored area

These measurements can later be visualized through a dashboard.

---

## 10. Data Quality Issues

Several factors can affect the quality of the results:

### Occlusion

When one person blocks another person, detection or tracking can
become difficult.

### Poor Lighting

Low-light conditions can reduce detection quality.

### Motion Blur

Fast movement can make people harder to detect and track.

### Camera Movement

A moving camera can affect tracking stability.

### High Crowd Density

Large groups of people can increase the difficulty of maintaining
individual tracking IDs.

### Incorrect Counting Line

An incorrectly positioned counting line can result in incorrect
entry or exit counts.

---

## 11. Limitations

The current system has several limitations:

- Counting accuracy depends on video quality.
- Heavy occlusion can cause tracking errors.
- Very crowded scenes can reduce tracking reliability.
- Camera angle affects detection and counting.
- The system does not identify individual people.
- The system does not perform facial recognition.
- Footfall counts may contain errors in difficult scenes.

---

## 12. Evaluation Metrics

The system can be evaluated using:

- Detection accuracy
- Tracking stability
- Entry counting accuracy
- Exit counting accuracy
- Processing speed / FPS
- Number of missed detections
- Number of incorrect detections

These metrics can be measured using test videos with known or
manually verified counts.

---

## 13. Future Analysis

Future versions of the system can analyze:

- Footfall by time period
- Peak visitor periods
- Hourly entry and exit trends
- Average occupancy
- Visitor flow
- Area-wise movement
- Historical footfall patterns

This information can be presented through an analytics dashboard.