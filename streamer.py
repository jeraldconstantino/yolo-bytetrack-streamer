import cv2
import random
from loguru import logger
from ultralytics import YOLO


class YoloByteTrackStreamer:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        model_path (str): path or name of the YOLO model (e.g. 'yolov8n.pt', 'yolov8x.pt')
        device (str): 'cpu' or 'cuda'
        """
        self.window_name = "ShopSight Object Detection and Tracking"
        self.model = YOLO(model_path)
        self.device = device
        self.track_id_colors = {}  # persistent across frames

    def _get_video_properties(self, source: int | str) -> tuple[float, int, int]:
        """Open the source once to get fps, width, height."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Error opening video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:  # NaN or 0
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, width, height

    def _get_color_for_track(self, track_id: int) -> tuple[int, int, int]:
        """Return a consistent random color for each track_id."""
        if track_id not in self.track_id_colors:
            self.track_id_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        return self.track_id_colors[track_id]

    def run(self, 
            source: int | str, 
            target_classes=None, 
            output_path=None):
        """
        source:
            int -> camera index (0 = default webcam)
            str -> video path (e.g., 'video.mp4') or URL (RTSP/HTTP)
        target_classes:
            list of COCO class IDs to keep (e.g., [0, 32]) or None for all.
        output_path:
            if not None, saves the annotated video to this file path.
        """
        logger.info(f"Starting stream from source: {source}")

        # Get video properties for optional writer
        fps, width, height = self._get_video_properties(source)

        # Setup video writer if output path is not None
        out = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset colors for each run
        self.track_id_colors = {}

        # Start tracking stream
        results = self.model.track(
            source=source,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            classes=target_classes,
            device=self.device,
        )

        for frame_id, result in enumerate(results):
            frame = result.orig_img.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                ids = result.boxes.id
                if ids is not None:
                    track_ids = ids.int().cpu().tolist()
                else:
                    # If tracker didn't assign IDs, use -1 for "untracked"
                    track_ids = [-1] * len(bboxes)

                for track_id, bbox, cls_id in zip(track_ids, bboxes, class_ids):
                    # Filter by target classes if specified
                    if target_classes is not None and cls_id not in target_classes:
                        continue

                    color = self._get_color_for_track(track_id)
                    x1, y1, x2, y2 = map(int, bbox)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # Label text
                    text = f"ID:{track_id} CLS:{cls_id}"
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )

                    bg_x1 = x1
                    bg_y1 = y1 - 10 - th
                    bg_x2 = x1 + tw
                    bg_y2 = y1 - 10 + baseline

                    bg_y1 = max(0, bg_y1)
                    bg_y2 = min(frame.shape[0], bg_y2)

                    # White background for text
                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                    cv2.putText(
                        frame, text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2
                    )

            # Show frame
            cv2.imshow(self.window_name, frame)

            # Save if writer is enabled
            if out is not None:
                out.write(frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out is not None:
            out.release()
            logger.info(f"Output video saved to {output_path}")

        cv2.destroyAllWindows()
        logger.info("Stream ended.")
