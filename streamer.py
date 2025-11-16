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

    def _draw_bounding_box(self, 
                           frame,
                           bbox,
                           class_name: str,
                           track_id: int,
                           conf: float,
                           color: tuple[int, int, int]):
        """
        Draws a solid bounding box with a colored label bar:
        '#ID class_name conf'
        """

        x1, y1, x2, y2 = map(int, bbox)

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label text
        label = f"#{track_id} {class_name} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        label_thickness = 2

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)

        # label background
        label_x1 = x1
        label_y2 = max(y1, th + 8)
        label_y1 = label_y2 - th - 8
        label_x2 = x1 + tw + 8

        # filled rectangle (same as box color)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # Dynamic text color based on luminance 
        R, G, B = color
        luminance = 0.299 * R + 0.587 * G + 0.114 * B
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        # Draw text
        cv2.putText(
            frame,
            label,
            (label_x1 + 4, label_y2 - 4),
            font,
            font_scale,
            text_color,
            label_thickness,
            cv2.LINE_AA,
        )

    def _draw_counts_panel(self, frame, class_counts: dict[str, int]):
        """
        Draws a semi-transparent panel in the upper-right with per-class counts,
        e.g.:
            Objects
            person: 3
            car: 2
        """
        if not class_counts:
            return

        h, w, _ = frame.shape

        # build lines: header + each class
        lines = ["Objects"]
        for name, cnt in class_counts.items():
            lines.append(f"{name}: {cnt}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        margin = 10
        line_spacing = 6

        # compute panel size
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
        ]
        max_text_w = max(ts[0] for ts in text_sizes)
        text_h = max(ts[1] for ts in text_sizes)

        panel_w = max_text_w + 2 * margin
        panel_h = len(lines) * (text_h + line_spacing) + 2 * margin - line_spacing

        x2 = w - 10
        x1 = x2 - panel_w
        y1 = 10
        y2 = y1 + panel_h

        # semi-transparent dark background
        overlay = frame.copy()
        panel_color = (0, 0, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), panel_color, -1)
        alpha = 0.4
        frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # draw text lines
        y = y1 + margin + text_h
        for i, line in enumerate(lines):
            if i == 0:
                color = (0, 255, 255)  # header color
            else:
                color = (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (x1 + margin, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += text_h + line_spacing

    def run(
        self,
        source: int | str,
        target_classes = None,
        output_path = None,
        show_window: bool | None = None,
    ):
        """
        source:
            int -> camera index (0 = default webcam)
            str -> video path (e.g., 'video.mp4') or URL (RTSP/HTTP)
        target_classes:
            list of COCO class IDs to keep (e.g., [0, 32]) or None for all.
        output_path:
            if not None, saves the annotated video to this file path.
        show_window: bool | None
            If None (default), decide automatically:
                - camera (int)      -> True  (show)
                - video file (.mp4) -> False (headless)
            If True/False, override behavior.
        """
        logger.info(f"Starting stream from source: {source}")

        if show_window is None:
            if isinstance(source, int):
                show_window_effective = True           # webcam → show
            elif isinstance(source, str) and source.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):
                show_window_effective = False          # video file → headless
            else:
                show_window_effective = True           # RTSP/HTTP/etc → show
        else:
            show_window_effective = show_window

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

            # per-frame counts
            class_counts: dict[str, int] = {}

            if result.boxes is not None and len(result.boxes) > 0:
                bboxes = result.boxes.xyxy.cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()

                ids = result.boxes.id
                if ids is not None:
                    track_ids = ids.int().cpu().tolist()
                else:
                    track_ids = [-1] * len(bboxes)

                for track_id, bbox, cls_id, conf in zip(
                    track_ids, bboxes, class_ids, confs
                ):
                    if target_classes is not None and cls_id not in target_classes:
                        continue

                    class_name = self.model.names[int(cls_id)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    color = self._get_color_for_track(track_id)

                    self._draw_bounding_box(
                        frame=frame,
                        bbox=bbox,
                        class_name=class_name,
                        track_id=track_id,
                        conf=conf,
                        color=color,
                    )

            # draw upper-right counts panel
            self._draw_counts_panel(frame, class_counts)

            # Only show window if enabled
            if show_window_effective:
                cv2.imshow(self.window_name, frame)
                # allow user to quit with 'q' only when window exists
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # headless mode: no imshow, no popup
                pass

            # Save if writer is enabled
            if out is not None:
                out.write(frame)

        if out is not None:
            out.release()
            logger.info(f"Output video saved to {output_path}")

        if show_window_effective:
            cv2.destroyAllWindows()

        logger.info("Stream ended.")