# Engineering Decision Log

This document records important technical decisions made during the
development of the AI-Powered Footfall Analytics & Visitor Monitoring
System.

---

## Decision 1 — Use Python

### Decision

Use Python as the primary programming language.

### Reason

Python provides strong support for computer vision, AI/ML, data
processing, and rapid application development.

The project also uses libraries such as OpenCV and Ultralytics that
provide Python interfaces.

### Alternative Considered

C++

### Why Rejected

C++ can provide high performance, but Python provides a simpler
development workflow and is sufficient for the current project
requirements.

---

## Decision 2 — Use YOLOv8 for Person Detection

### Decision

Use YOLOv8 for detecting people in video frames.

### Reason

The project requires object detection from video frames. YOLOv8
provides the bounding-box information required by the tracking and
footfall-counting stages.

### Alternative Considered

Traditional image-processing techniques.

### Why Rejected

Traditional approaches based on manually designed image-processing
rules can be less robust for different backgrounds, lighting
conditions, and people appearances.

---

## Decision 3 — Use ByteTrack for Object Tracking

### Decision

Use ByteTrack for multi-object tracking.

### Reason

The system needs to maintain tracking identities across consecutive
video frames.

Tracking IDs are required to prevent the same person from being
counted repeatedly.

### Alternative Considered

Frame-by-frame detection without tracking.

### Why Rejected

Independent frame detection does not maintain object identity and
could result in repeated counting of the same person.

---

## Decision 4 — Use OpenCV for Video Processing

### Decision

Use OpenCV for reading, processing, displaying, and writing video
frames.

### Reason

OpenCV provides the required functionality for video capture and
frame-level computer-vision processing.

### Alternative Considered

A specialized video-processing framework.

### Why Rejected

The additional complexity was not necessary for the current
application.

---

## Decision 5 — Use a Virtual Counting Line

### Decision

Use a predefined virtual line to determine entry and exit events.

### Reason

The primary objective is to determine when a tracked person crosses
a defined boundary.

A line-crossing approach is simple, understandable, and appropriate
for a controlled camera view.

### Alternative Considered

Zone-based occupancy estimation.

### Why Rejected

Zone-based analysis would add complexity that is not required for the
current entry/exit counting objective.

---

## Decision 6 — Do Not Use RAG

### Decision

Do not implement Retrieval-Augmented Generation in the core system.

### Reason

The system processes video rather than documents or textual knowledge.

The main task is visual detection, tracking, movement analysis, and
footfall counting.

### Alternative Considered

Document retrieval using a vector database and an LLM.

### Why Rejected

There is no core requirement for retrieving external textual
knowledge. Adding RAG would increase system complexity without
providing significant value to the current problem.

---

## Decision 7 — Do Not Use an AI Agent

### Decision

Do not implement an autonomous AI Agent in the core system.

### Reason

The current system follows a predefined computer-vision pipeline and
does not require autonomous tool selection or multi-step decision
making.

### Alternative Considered

An agentic workflow using tools and APIs.

### Why Rejected

An agent would not provide meaningful additional functionality for
the current footfall-counting problem.

---

## Decision 8 — Use Streamlit for the Application Layer

### Decision

Use Streamlit for the planned user-facing analytics application.

### Reason

The project requires a simple interface through which users can
upload/process video and view footfall results.

Streamlit can provide a practical dashboard without requiring a large
frontend architecture.

### Alternative Considered

React-based frontend.

### Why Rejected

A React frontend would require additional frontend and backend
integration. Streamlit is more suitable for the current project scope
and rapid AI/ML application development.

---

## Decision 9 — Use Docker for Reproducibility

### Decision

Package the application using Docker.

### Reason

Docker will make the application environment more reproducible and
will reduce dependency-related differences between development and
deployment environments.

### Alternative Considered

Manual environment setup.

### Why Rejected

Manual setup can lead to dependency and environment inconsistencies.

---

## Decision 10 — Add Automated Testing

### Decision

Add basic automated tests for important application components.

### Reason

Testing helps verify that counting logic, input validation, and
application functionality behave as expected.

### Alternative Considered

Only manual testing.

### Why Rejected

Manual testing alone does not provide a repeatable way to verify
important functionality after code changes.

---

## Summary

The project follows a principle of selecting technologies based on
actual technical requirements.

The core architecture therefore focuses on:

Python
+
OpenCV
+
YOLOv8
+
ByteTrack
+
Footfall Counting

Additional technologies such as Streamlit, Docker, and automated
testing are introduced to improve application usability,
reproducibility, and engineering reliability.