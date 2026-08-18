import os
import streamlit as st

from footfall_processor import process_video
from video_validator import validate_video


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Footfall Analytics",
    page_icon="👥",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("👥 AI-Powered Footfall Analytics")

st.markdown(
    """
    ### Computer Vision Based Visitor Monitoring System

    Detect, track and analyze people from video using:

    **YOLOv8 + ByteTrack + OpenCV**
    """
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Configuration")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

device = st.sidebar.selectbox(
    "Processing Device",
    ["cpu"]
)

st.sidebar.divider()

st.sidebar.subheader("🧠 AI Pipeline")

st.sidebar.write(
    "Video → YOLOv8 → ByteTrack → "
    "Movement Analysis → IN/OUT Counting"
)

st.sidebar.divider()

st.sidebar.subheader("📌 Model")

st.sidebar.write("YOLOv8 Nano")
st.sidebar.write("ByteTrack Tracker")

st.sidebar.divider()

st.sidebar.info(
    "Automated footfall monitoring and "
    "visitor movement analysis."
)


# ==========================================
# PROJECT OVERVIEW
# ==========================================

st.subheader("📋 Project Overview")

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        ### 🎯 Person Detection

        YOLOv8 detects people in each
        video frame.
        """
    )

with col2:

    st.markdown(
        """
        ### 🆔 Object Tracking

        ByteTrack maintains tracking
        identities across frames.
        """
    )

with col3:

    st.markdown(
        """
        ### 📊 Footfall Counting

        Virtual line crossing is used
        to calculate IN and OUT counts.
        """
    )


st.divider()


# ==========================================
# VIDEO UPLOAD
# ==========================================

st.subheader("🎥 Upload Video")

uploaded_file = st.file_uploader(
    "Select a video for footfall analysis",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ==========================================
# PROCESS UPLOADED VIDEO
# ==========================================

if uploaded_file is not None:

    st.success(
        f"Video selected: {uploaded_file.name}"
    )

    # ======================================
    # SAVE UPLOADED VIDEO
    # ======================================

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    input_path = os.path.join(
        "output",
        f"uploaded_input{extension}"
    )

    os.makedirs(
        "output",
        exist_ok=True
    )

    with open(
        input_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    # ======================================
    # VALIDATE VIDEO
    # ======================================

    is_valid, validation_message = (
        validate_video(input_path)
    )

    if not is_valid:

        st.error(
            f"❌ Invalid Video: {validation_message}"
        )

        st.stop()

    st.success(
        f"✅ {validation_message}"
    )

    # ======================================
    # INPUT VIDEO
    # ======================================

    st.subheader("🎬 Input Video")

    st.video(
        uploaded_file
    )

    st.divider()

    # ======================================
    # START PROCESSING
    # ======================================

    if st.button(
        "🚀 Start Footfall Analysis",
        type="primary",
        use_container_width=True
    ):

        output_path = os.path.join(
            "output",
            "streamlit_output.mp4"
        )

        progress = st.progress(0)

        status = st.empty()

        status.info(
            "Initializing YOLOv8 and ByteTrack..."
        )

        try:

            # ==================================
            # PROCESS VIDEO
            # ==================================

            result = process_video(
                video_source=input_path,
                output_path=output_path,
                conf_threshold=confidence,
                device=device,
                show_window=False
            )

            progress.progress(100)

            status.success(
                "✅ Footfall analysis completed!"
            )

            st.divider()

            # ==================================
            # FOOTFALL RESULTS
            # ==================================

            st.subheader(
                "📊 Footfall Results"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    label="🟢 IN",
                    value=result["in_count"]
                )

            with col2:

                st.metric(
                    label="🔴 OUT",
                    value=result["out_count"]
                )

            with col3:

                st.metric(
                    label="👥 Total Footfall",
                    value=result["total_footfall"]
                )

            st.divider()

            # ==================================
            # PERFORMANCE
            # ==================================

            st.subheader(
                "⚡ Processing Performance"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Frames Processed",
                    result["frames_processed"]
                )

            with col2:

                st.metric(
                    "Average FPS",
                    round(
                        result["average_fps"],
                        2
                    )
                )

            with col3:

                st.metric(
                    "Processing Time",
                    f"{result['processing_time']:.2f}s"
                )

            st.divider()

            # ==================================
            # STEP COUNTING
            # ==================================

            st.subheader(
                "🚶 Step Count"
            )

            step_data = result[
                "step_counts"
            ]

            if step_data:

                for person_id, steps in step_data.items():

                    st.write(
                        f"Person ID **{person_id}** → "
                        f"**{steps} steps**"
                    )

            else:

                st.info(
                    "No step-count data available."
                )

            st.divider()

            # ==================================
            # OUTPUT VIDEO
            # ==================================

            if os.path.exists(
                result["output_path"]
            ):

                st.subheader(
                    "🎥 Processed Video"
                )

                with open(
                    result["output_path"],
                    "rb"
                ) as video_file:

                    video_data = (
                        video_file.read()
                    )

                st.video(
                    video_data
                )

                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_data,
                    file_name="footfall_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

        except Exception as error:

            progress.empty()

            status.error(
                "❌ Processing failed."
            )

            st.exception(
                error
            )

else:

    st.info(
        "👆 Upload a video above to start "
        "the footfall analysis."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI-Powered Footfall Analytics | "
    "YOLOv8 + ByteTrack + OpenCV"
)
