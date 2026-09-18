import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing

class HandLandmarkDrawer:
    @staticmethod
    def draw_landmarks(image, detection_result, hand_indices=None, copy=True):
        annotated_image = np.copy(image) if copy else image
        if detection_result and hasattr(detection_result, "hand_landmarks") and detection_result.hand_landmarks:
            indices = hand_indices if hand_indices is not None else range(len(detection_result.hand_landmarks))
            for i in indices:
                hand_landmarks = detection_result.hand_landmarks[i]
                hand_proto = landmark_pb2.NormalizedLandmarkList()
                hand_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                    for lm in hand_landmarks
                ])
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        return annotated_image

class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", camera_index=0, use_gpu=True):
        self.model_path = model_path
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        self._rgb = None      # reusable detection buffer, sized on first frame
        self._last_ts = 0     # monotonic watermark for MediaPipe timestamps
        self._init_landmarker(use_gpu)

    def _init_landmarker(self, use_gpu):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        delegate = BaseOptions.Delegate.GPU if use_gpu else None
        base_opts = BaseOptions(model_asset_path=self.model_path, delegate=delegate) if delegate else BaseOptions(model_asset_path=self.model_path)
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def _next_timestamp(self):
        """Strictly-increasing millisecond timestamp.

        MediaPipe's VIDEO running mode rejects a timestamp that is not greater
        than the previous one, which wall-clock milliseconds hit as soon as two
        frames land inside the same millisecond.
        """
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def _reopen_camera(self):
        """Retry a camera that failed to open; returns whether it's open now.

        On first launch macOS hasn't granted camera access yet, so OpenCV asks
        for it and fails the open. Retrying picks the camera up once the user
        clicks Allow, instead of needing a restart. The pause paces the retries
        and keeps the gesture loop from spinning while it waits.
        """
        time.sleep(1.0)
        return self.cap.open(self.camera_index)

    def get_frame_and_landmarks(self, active_area_ratio=1.0):
        if not self.cap.isOpened() and not self._reopen_camera():
            return None, None
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)

        # Convert straight into a reused RGB buffer and mask the excluded strip
        # there. Detection then costs one 3-channel conversion per frame rather
        # than a full-frame copy plus a 4-channel one.
        h, w = frame.shape[:2]
        if self._rgb is None or self._rgb.shape[:2] != (h, w):
            self._rgb = np.empty((h, w, 3), dtype=np.uint8)
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb)
        exclusion_y = int(h * active_area_ratio)
        if exclusion_y < h:
            self._rgb[exclusion_y:, :] = 0

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self._rgb)
        results = self.landmarker.detect_for_video(mp_image, self._next_timestamp())
        return frame, results

    def close(self):
        self.cap.release()
        try:
            self.landmarker.close()
        except Exception:
            pass
