# Data & Knowledge Strategy

## 1. Purpose

The Footfall Analytics System requires video data as its primary input.
The video is processed frame-by-frame to detect and track people and
calculate entry and exit counts.

The system does not require a textual knowledge base or external
documents for its core footfall-counting functionality.

---

## 2. Data Sources

The system supports two primary input sources:

### A. Video File

A recorded video can be provided as an input to the system.

Example:

- MP4 video
- Recorded surveillance or demonstration footage
- User-provided video

### B. Webcam

The system can process a live webcam stream.

The webcam provides continuous video frames that are processed by the
computer-vision pipeline.

---

## 3. Data Format

The primary data format is video.

Supported input can include:

- `.mp4`
- Webcam video stream

The video is internally processed as a sequence of individual frames.

Processing flow:

Video
↓
Frames
↓
Person Detection
↓
Bounding Boxes
↓
Object Tracking
↓
Tracking IDs
↓
Movement Analysis
↓
Entry / Exit Counts

---

## 4. Data Requirements

For reliable operation, the input video should preferably have:

- Clearly visible people
- Sufficient lighting
- Reasonable video resolution
- A stable camera position
- A clearly defined counting area
- Limited extreme motion blur

The system is designed to detect people using a computer-vision
object detection model and track them across consecutive frames.

---

## 5. Data Processing

The input video is processed using Python and OpenCV.

Each frame is passed to the YOLOv8 object detection model.

The system identifies people and obtains their bounding boxes.

ByteTrack is then used to maintain tracking identities across frames.

The center position of each tracked person is used by the counting
logic to determine whether a person has crossed the configured
counting line.

---

## 6. Data Quality Considerations

The quality of the input video can affect detection and tracking
performance.

Potential factors include:

- Poor lighting
- Low video resolution
- Motion blur
- Heavy crowding
- People blocking one another
- Camera movement
- Partial visibility of people
- Incorrect counting-line placement

These conditions can cause missed detections, incorrect tracking or
incorrect entry/exit counts.

---

## 7. Privacy and Security

This project is intended for person detection and footfall analysis.

The system does not perform facial recognition or attempt to determine
the identity of individual people.

The system primarily uses:

- Person bounding boxes
- Tracking IDs
- Centroid positions
- Movement information
- Entry/exit counts

Video data should only be collected and processed when appropriate
permission has been obtained.

Where possible, video processing can be performed locally so that
sensitive video does not need to be uploaded to an external service.

---

## 8. Data Storage

The system can generate processed video output for demonstration and
analysis.

Generated files should be stored locally in the project output
directory.

Large video files and unnecessary datasets should not be committed
to the GitHub repository unless there is a specific reason to include
them.

---

## 9. Why External Knowledge / RAG Is Not Required

The core footfall-counting problem does not require retrieval of
external documents or textual knowledge.

The primary input is visual data, and the required output is generated
through computer-vision detection, tracking and counting.

Therefore, a Retrieval-Augmented Generation (RAG) system is not required
for the core implementation.