# AI Feature Decisions

## 1. Overview

The project was evaluated for the use of modern AI technologies such
as Retrieval-Augmented Generation (RAG), Large Language Models (LLMs),
and AI Agents.

The decision was made to use only technologies that provide clear
technical value to the footfall-counting problem.

The core system therefore focuses on computer vision, object detection,
multi-object tracking, and movement analysis.

---

# 2. RAG Decision

## Decision

RAG is not implemented in the core footfall-counting system.

## Reason

The main input to the application is visual video data.

The core problem requires the system to:

- Detect people
- Track people
- Analyze movement
- Detect line crossings
- Calculate entry and exit counts

These operations do not require retrieval of external documents or
textual knowledge.

A typical RAG pipeline would be:

Documents
↓
Chunking
↓
Embeddings
↓
Vector Database
↓
Retrieval
↓
LLM
↓
Response

This architecture does not provide significant value for the core
footfall-counting task.

## Conclusion

RAG was intentionally not added because it would introduce additional
complexity without solving an actual requirement of the core system.

---

# 3. LLM Decision

## Decision

An LLM is not required for the core implementation.

## Reason

The main processing pipeline is visual and computational:

Video
↓
Object Detection
↓
Object Tracking
↓
Movement Analysis
↓
Footfall Count

An LLM is primarily designed for language-based tasks and is therefore
not necessary for the core detection and tracking workflow.

## Possible Future Use

An LLM could be introduced in a future version for features such as:

- Natural-language analytics
- Automated daily footfall reports
- Business insights
- Natural-language queries about historical footfall
- Explanation of unusual visitor patterns

These features are outside the scope of the current core system.

---

# 4. AI Agent Decision

## Decision

An autonomous AI Agent is not implemented in the core system.

## Reason

The current application follows a predefined computer-vision pipeline.

The processing flow is:

Input Video
↓
YOLOv8
↓
ByteTrack
↓
Movement Analysis
↓
Footfall Counting
↓
Output

The system does not need an autonomous component to decide which
external tools to call, perform multi-step tasks, or independently
execute business actions.

Therefore, adding an AI Agent would increase system complexity without
providing a necessary capability.

---

# 5. Agentic AI Future Possibility

An agentic component could become useful in a future version if the
system is expanded into a complete business analytics platform.

For example:

User
↓
AI Agent
↓
Footfall Database
↓
Analytics Tool
↓
Trend Analysis
↓
Report Generation
↓
User Response

Possible agent capabilities could include:

- Querying historical footfall data
- Comparing different time periods
- Detecting unusual traffic patterns
- Generating reports
- Calling analytics APIs
- Answering business questions

These capabilities are considered future improvements rather than
requirements of the current computer-vision system.

---

# 6. Engineering Principle

The project follows the principle of using technology only when it
solves a real problem.

The system therefore prioritizes:

- YOLOv8
- ByteTrack
- OpenCV
- Python
- Computer vision
- Footfall analytics

instead of adding technologies only to increase the technology count.

This keeps the architecture simpler, easier to test, and easier to
explain and maintain.

---

# 7. Final Decision

| Technology | Decision | Reason |
|------------|----------|--------|
| YOLOv8 | Used | Person detection |
| ByteTrack | Used | Multi-object tracking |
| OpenCV | Used | Video processing |
| LLM | Not used in core | Not required for visual processing |
| RAG | Not used | No external knowledge retrieval requirement |
| AI Agent | Not used | No autonomous multi-step action required |
| Streamlit | Planned | User-facing analytics application |
| FastAPI | Planned | API exposure where appropriate |
| Docker | Planned | Reproducible deployment |