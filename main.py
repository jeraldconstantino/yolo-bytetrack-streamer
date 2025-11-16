from streamer import YoloByteTrackStreamer

streamer = YoloByteTrackStreamer(
    model_path="yolov8n.pt",  # or "yolov8x.pt"
    device="cpu"
)

SELECTED_CLASS_IDS = [0, 32]  # person, sports ball
streamer.run(source=0, target_classes=SELECTED_CLASS_IDS)