import cv2
import os
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

    def _draw_bounding_box(self, frame, bbox, class_name, track_id, conf, color):
        """Draw a semi-transparent box with a nice label bar."""
        x1, y1, x2, y2 = map(int, bbox)

        # semi-transparent filled box overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # filled rect
        alpha = 0.2  # transparency factor
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # thinner solid border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label text (class name + confidence + track ID)
        label = f"{class_name} {conf:.2f}  ID:{track_id}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # label bar above the box
        label_x1 = x1
        label_y1 = max(0, y1 - th - 8)
        label_x2 = x1 + tw + 8
        label_y2 = y1

        # darker version of box color for label background
        bg_color = (int(color[0] * 0.4), int(color[1] * 0.4), int(color[2] * 0.4))
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)

        # white text on top of colored bar
        cv2.putText(
            frame,
            label,
            (label_x1 + 4, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

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
            output_path = os.path.abspath(output_path)
            out_dir = os.path.dirname(output_path)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error(f"Failed to open VideoWriter for: {output_path}")
                out = None
            else:
                logger.info(f"Output video will be saved to: {output_path}")

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

            # Reset counts for this frame
            class_counts: dict[str, int] = {}

            if result.boxes is not None and len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()

                ids = result.boxes.id
                if ids is not None:
                    track_ids = ids.int().cpu().tolist()
                else:
                    # If tracker didn't assign IDs, use -1 for "untracked"
                    track_ids = [-1] * len(bboxes)

                for track_id, bbox, cls_id, conf in zip(track_ids, bboxes, class_ids, confs):
                    # Filter by target classes if specified
                    if target_classes is not None and cls_id not in target_classes:
                        continue

                    # Count this object by its class name
                    class_name = self.model.names[int(cls_id)]  # Convert class ID → class name
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    # Draw bounding box + label
                    color = self._get_color_for_track(track_id)
                    self._draw_bounding_box(
                        frame=frame,
                        bbox=bbox,
                        class_name=class_name,
                        track_id=track_id,
                        conf=conf,
                        color=color,
                    )

            # Draw class counts in upper-left corner
            if class_counts:
                y0 = 30
                dy = 22
                for i, (name, count) in enumerate(class_counts.items()):
                    text = f"{name}: {count}"
                    cv2.putText(
                        frame,
                        text,
                        (10, y0 + i * dy),  # (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),  # green text
                        2,
                        cv2.LINE_AA,
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
