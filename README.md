# AI-Powered Footfall Counting System

An AI-powered computer vision system for detecting, tracking,
and counting people entering and leaving a monitored area.

## Features

- Person detection using YOLOv8
- Multi-object tracking using ByteTrack
- IN/OUT footfall counting
- Step counting
- Video input validation
- Streamlit web interface
- Processing performance metrics
- Automated testing using Pytest
- Processed video generation

## System Architecture

Video Input
    ↓
Video Validation
    ↓
YOLOv8 Detection
    ↓
ByteTrack Tracking
    ↓
Centroid / Movement Analysis
    ↓
Virtual Counting Line
    ↓
IN / OUT Counting
    ↓
Analytics & Processed Video

## Tech Stack

- Python
- OpenCV
- YOLOv8
- ByteTrack
- NumPy
- Streamlit
- Pytest

## Project Structure

```text
Footfall-Counting-System/
├── app.py
├── main.py
├── footfall_processor.py
├── video_validator.py
├── config/
├── tests/
├── docs/
├── screenshots/
├── output/
├── requirements.txt
├── .gitignore
└── README.md
