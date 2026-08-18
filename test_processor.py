from footfall_processor import process_video


VIDEO_PATH = "Sanjivani.mp4"

result = process_video(
    video_source=VIDEO_PATH,
    model_path="yolov8n.pt",
    output_path="output/test_output.mp4",
    conf_threshold=0.25,
    device="cpu",
    show_window=True
)

print("\n" + "=" * 50)
print("PROCESSING COMPLETE")
print("=" * 50)

print("Frames Processed:", result["frames_processed"])
print("IN:", result["in_count"])
print("OUT:", result["out_count"])
print("Total Footfall:", result["total_footfall"])
print("Output:", result["output_path"])

print(
    "Processing Time:",
    round(result["processing_time"], 2),
    "seconds"
)

print(
    "Average FPS:",
    round(result["average_fps"], 2)
)

print("=" * 50) 
