# ==========================================
# Footfall Counting System Configuration
# ==========================================

# -----------------------------
# MODEL SETTINGS
# -----------------------------

MODEL_PATH = "yolov8n.pt"

# Minimum confidence for person detection
CONF_THRESHOLD = 0.25

# Current system uses CPU
DEVICE = "cpu"


# -----------------------------
# VIDEO SETTINGS
# -----------------------------

VIDEO_FILE_PATH = "Sanjivani.mp4"

# Output video
OUTPUT_PATH = "output/footfall_output.mp4"


# -----------------------------
# TRACKING SETTINGS
# -----------------------------

TRACKER_CONFIG = "bytetrack.yaml"


# -----------------------------
# STEP COUNTING SETTINGS
# -----------------------------

STEP_THRESHOLD = 4

# Prevent repeated step counting
COOLDOWN_FRAMES = 5


# -----------------------------
# TRAJECTORY SETTINGS
# -----------------------------

# Number of previous positions
# stored for each tracked person
MAX_TRAJECTORY_POINTS = 20