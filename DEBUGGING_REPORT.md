# Debugging Report

This document records meaningful technical problems encountered during
the development of the AI-Powered Footfall Analytics & Visitor
Monitoring System.

---

## Problem 1 — Slow Video Processing

### What Failed?

The video opened successfully, but the Footfall Counting System processed
the video very slowly. The output was not being processed at a suitable
speed for smooth video analysis.

### Error / Problem Observed

There was no application crash or major error message. The main issue
was low processing speed.

The system was able to:

- Open the input video
- Read video frames
- Detect people
- Track people
- Generate the output

However, the processing speed was slow and the video did not run
smoothly in real time.

### Investigation

The processing pipeline was examined to identify the components that
could affect execution speed.

The main processing stages were:

Input Video
↓
OpenCV Frame Processing
↓
YOLOv8 Person Detection
↓
ByteTrack Tracking
↓
Movement Analysis
↓
Output Video

YOLOv8 inference and frame-by-frame processing were identified as the
main areas requiring performance consideration.

### Solution Implemented

The system was evaluated with a focus on improving processing
efficiency.

The project uses a lightweight YOLO model configuration suitable for
video processing and avoids unnecessary processing outside the main
detection and tracking pipeline.

Further performance optimization will be evaluated using processing
speed/FPS measurements.

### Verification

The application was executed again using the input video.

The video opened correctly and the complete detection and tracking
pipeline executed successfully.

Processing speed will be measured using FPS in a later performance
evaluation stage to quantitatively compare improvements.


## Problem 2 — Incorrect Person Detection and IN/OUT Counting

### What Failed?

In some video frames, the system did not detect people correctly.
Because the footfall-counting logic depends on person detection and
tracking, incorrect detections sometimes resulted in incorrect IN/OUT
counts.

### Error / Problem Observed

The system was able to process the video, but in certain situations:

- A person could be missed by the detector.
- A person's tracking could become unstable.
- A person crossing the counting line could sometimes be counted
  incorrectly.
- Difficult scenes could affect the final IN/OUT count.

### Investigation

The complete detection and tracking pipeline was examined:

Video Frame
↓
YOLOv8 Person Detection
↓
ByteTrack Tracking
↓
Centroid Calculation
↓
Counting Line Detection
↓
IN / OUT Count

The issue was investigated by observing the detected bounding boxes,
tracking IDs, centroid movement, and the position of people relative
to the counting line.

### Solution Implemented

The detection and tracking configuration was reviewed to make sure
that only person detections were used by the counting logic.

The counting logic was also reviewed so that an IN or OUT event is
generated based on the movement of a tracked person's centroid across
the defined counting line.

The system was tested with different movements to check whether
people were being detected and counted consistently.

### Verification

The system was executed again using the input video and the detection,
tracking, and counting results were observed.

The output was checked to verify that:

- People were detected with bounding boxes.
- Tracking IDs were maintained across frames.
- The counting line was correctly displayed.
- IN/OUT events were generated when tracked people crossed the line.

Further quantitative evaluation of counting accuracy will be performed
using manually verified test videos.