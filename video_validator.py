import cv2
import os


SUPPORTED_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


def validate_video(video_path):
    """
    Validate whether a video file exists,
    has a supported extension, and can be opened.
    """

    # Check file existence
    if not os.path.exists(video_path):
        return False, "Video file does not exist."

    # Check file extension
    extension = os.path.splitext(
        video_path
    )[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return False, (
            f"Unsupported video format: {extension}"
        )

    # Try opening the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return False, "Video could not be opened."

    # Check video properties
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if width <= 0 or height <= 0:
        return False, "Invalid video dimensions."

    if frame_count <= 0:
        return False, "Video contains no frames."

    return True, "Video is valid."