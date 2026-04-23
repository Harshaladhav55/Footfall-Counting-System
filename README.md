📌 Footfall Counting System
🔹 Overview

The Footfall Counting System is a computer vision-based application that detects and counts the number of people passing through a specific area in a video stream. It uses object detection techniques to track human movement and generate accurate footfall data in real-time or from recorded videos.

This system can be used in places like malls, universities, offices, and public areas to analyze crowd density and movement patterns.

🔹 Key Features
👤 Real-time person detection using YOLO
🎥 Works with both live camera and recorded videos
🔢 Accurate footfall counting
📊 Output video with detection visualization
⚡ Fast and efficient processing
🧠 Easy to use and scalable
🔹 Tech Stack
Programming Language: Python
Libraries & Frameworks:
OpenCV
NumPy
Ultralytics YOLO (YOLOv8)
Tools:
Visual Studio Code
Git
GitHub
🔹 System Architecture
Input Video / Camera Feed
Frame Extraction
Object Detection (YOLO Model)
Tracking & Counting Logic
Output Generation (Video + Count)
🔹 Project Architecture
FOOTFALL-COUNTING-SYSTEM/
│
├── models/              # YOLO model files (.pt)
├── output/              # Output videos
├── main.py              # Main execution script
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── .gitignore           # Ignored files
🔹 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/Footfall-Counting-System.git
cd Footfall-Counting-System
2️⃣ Create Virtual Environment
python -m venv venv
3️⃣ Activate Virtual Environment
Windows:
venv\Scripts\activate
4️⃣ Install Dependencies
pip install -r requirements.txt
5️⃣ Download Model File

Download YOLO model (yolov8n.pt) and place it inside:

models/
6️⃣ Run the Project
python main.py
🔹 Applications
🏬 Shopping malls (customer tracking)
🎓 Universities (student movement analysis)
🏢 Offices (attendance & flow monitoring)
🚉 Public transport stations
🎉 Event management
🔹 Future Scope
📈 Dashboard integration for analytics
☁️ Cloud-based processing
📡 IoT integration with sensors
📊 Real-time data visualization
🤖 Improved tracking with AI models
🔹 Author
Harshal Adhav
