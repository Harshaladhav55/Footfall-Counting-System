import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os
import time

from config.settings import (
    MODEL_PATH,
    CONF_THRESHOLD,
    DEVICE,
    OUTPUT_PATH,
    TRACKER_CONFIG,
    STEP_THRESHOLD,
    COOLDOWN_FRAMES,
    MAX_TRAJECTORY_POINTS
)


def process_video(
    video_source,
    model_path=MODEL_PATH,
    output_path=OUTPUT_PATH,
    conf_threshold=CONF_THRESHOLD,
    device=DEVICE,
    show_window=True
):
    """
    Process a video/webcam stream.

    Features:
    - YOLOv8 person detection
    - ByteTrack object tracking
    - IN/OUT footfall counting
    - Step counting
    - Person trajectory tracking
    - Performance measurement

    show_window=True:
        Display OpenCV window.

    show_window=False:
        Process video without opening an OpenCV window.
        Useful for Streamlit/API applications.
    """

    # ==========================================
    # LOAD MODEL
    # ==========================================

    model = YOLO(model_path)

    # ==========================================
    # OPEN VIDEO
    # ==========================================

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise ValueError(
            f"Cannot open video source: {video_source}"
        )

    # ==========================================
    # VIDEO PROPERTIES
    # ==========================================

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = int(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = 30

    # ==========================================
    # COUNTING LINE
    # ==========================================

    counting_line_x = width // 2

    # ==========================================
    # OUTPUT DIRECTORY
    # ==========================================

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # ==========================================
    # VIDEO WRITER
    # ==========================================

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # ==========================================
    # TRACKING VARIABLES
    # ==========================================

    states = {}

    in_count = 0
    out_count = 0

    track_history = defaultdict(list)

    step_counts = defaultdict(int)

    last_centroid_y = {}

    cooldown = defaultdict(int)

    frame_count = 0

    # ==========================================
    # PERFORMANCE TIMER
    # ==========================================

    start_time = time.time()

    # ==========================================
    # MAIN LOOP
    # ==========================================

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # ======================================
        # YOLO + BYTETRACK
        # ======================================

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            classes=[0],
            conf=conf_threshold,
            device=device,
            verbose=False
        )

        # ======================================
        # COUNTING LINE
        # ======================================

        cv2.line(
            frame,
            (counting_line_x, 0),
            (counting_line_x, height),
            (0, 255, 255),
            3
        )

        # ======================================
        # DETECTIONS
        # ======================================

        if results[0].boxes.id is not None:

            boxes = (
                results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                results[0]
                .boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, track_id in zip(
                boxes,
                track_ids
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                # ==================================
                # CENTROID
                # ==================================

                centroid_x = int(
                    (x1 + x2) / 2
                )

                centroid_y = int(
                    (y1 + y2) / 2
                )

                # ==================================
                # IN / OUT COUNTING
                # ==================================

                current_state = (
                    "left"
                    if centroid_x < counting_line_x
                    else "right"
                )

                if track_id not in states:

                    states[track_id] = current_state

                elif states[track_id] != current_state:

                    if current_state == "right":
                        in_count += 1
                    else:
                        out_count += 1

                    states[track_id] = current_state

                # ==================================
                # STEP COUNTING
                # ==================================

                cooldown[track_id] = max(
                    0,
                    cooldown[track_id] - 1
                )

                if (
                    track_id in last_centroid_y
                    and cooldown[track_id] == 0
                ):

                    delta_y = abs(
                        centroid_y -
                        last_centroid_y[track_id]
                    )

                    if delta_y > STEP_THRESHOLD:

                        step_counts[track_id] += 1

                        cooldown[track_id] = (
                            COOLDOWN_FRAMES
                        )

                last_centroid_y[
                    track_id
                ] = centroid_y

                # ==================================
                # TRAJECTORY
                # ==================================

                track_history[
                    track_id
                ].append(
                    (
                        centroid_x,
                        centroid_y
                    )
                )

                if (
                    len(
                        track_history[
                            track_id
                        ]
                    )
                    > MAX_TRAJECTORY_POINTS
                ):

                    track_history[
                        track_id
                    ].pop(0)

                # ==================================
                # DRAW PERSON
                # ==================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Steps:{step_counts[track_id]}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2
                )

                # ==================================
                # DRAW TRAJECTORY
                # ==================================

                for i in range(
                    1,
                    len(
                        track_history[
                            track_id
                        ]
                    )
                ):

                    pt1 = track_history[
                        track_id
                    ][i - 1]

                    pt2 = track_history[
                        track_id
                    ][i]

                    cv2.line(
                        frame,
                        pt1,
                        pt2,
                        (255, 0, 255),
                        2
                    )

        # ======================================
        # STATISTICS
        # ======================================

        cv2.putText(
            frame,
            f"IN: {in_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"OUT: {out_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"TOTAL: {in_count + out_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "YOLOv8 + ByteTrack + Step Counter",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ======================================
        # OPTIONAL OPENCV WINDOW
        # ======================================

        if show_window:

            cv2.imshow(
                "FootFall Counter",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):
                break

        # ======================================
        # SAVE FRAME
        # ======================================

        out.write(frame)

    # ==========================================
    # RELEASE RESOURCES
    # ==========================================

    cap.release()

    out.release()

    if show_window:
        cv2.destroyAllWindows()

    # ==========================================
    # PERFORMANCE
    # ==========================================

    end_time = time.time()

    processing_time = (
        end_time - start_time
    )

    if processing_time > 0:
        average_fps = (
            frame_count /
            processing_time
        )
    else:
        average_fps = 0

    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {
        "frames_processed": frame_count,
        "in_count": in_count,
        "out_count": out_count,
        "total_footfall": (
            in_count + out_count
        ),
        "step_counts": dict(
            step_counts
        ),
        "output_path": output_path,
        "processing_time": processing_time,
        "average_fps": average_fps
    }