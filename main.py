from streamer import YoloByteTrackStreamer

streamer = YoloByteTrackStreamer(
    model_path="models/yolov8m.pt",  # or "yolov8x.pt"
    device="cpu"
)

# SELECTED_CLASS_IDS = [0, 32]  # person, sports ball
# streamer.run(source=0, target_classes=SELECTED_CLASS_IDS)

SELECTED_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
streamer.run(source="data/input/vehicles.mp4", 
             target_classes=SELECTED_CLASS_IDS, 
             output_path="data/output/vehicles_tracked.mp4")