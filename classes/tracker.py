import cv2
import json
import math
import mediapipe as mp
import numpy as np
import subprocess
import sys
import threading
import time
from .mjpeg_capture import MJPEGCapture
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing

# How far to probe for capture devices.
_MAX_CAMERA_INDEX = 8

# AVFoundation numbers its cameras contiguously from 0, so the first miss is
# the end of the list and probing past it only makes OpenCV print "out device
# of bound" at the terminal. V4L2 leaves gaps (a metadata node at
# /dev/video1 between two real cameras), so there every index gets tried.
_CONTIGUOUS_CAMERA_INDICES = sys.platform == "darwin"


def _system_camera_names():
    """Camera names as macOS reports them, in its own order.

    Only used to label indices that probing already found, and only when the
    two counts agree -- system_profiler and OpenCV enumerate independently, so
    a mismatch (a virtual camera one sees and the other doesn't) means the
    positions can't be trusted to line up.
    """
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=5,
                             check=True).stdout
        items = json.loads(out).get("SPCameraDataType", [])
        return [it["_name"] for it in items
                if isinstance(it, dict) and it.get("_name")]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def list_cameras(in_use=None, max_index=_MAX_CAMERA_INDEX):
    """Selectable cameras, as [{'index': int, 'name': str}].

    Opening a device is the only portable way to learn it exists, so this
    costs the better part of a second per camera and belongs behind a cache,
    never in the gesture loop. `in_use` is reported without being touched:
    reopening the camera the loop is reading would fight it for the device.
    """
    indices = []
    for index in range(max_index):
        if index == in_use:
            indices.append(index)
            continue
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                indices.append(index)
            elif _CONTIGUOUS_CAMERA_INDICES:
                break
        finally:
            cap.release()

    names = _system_camera_names()
    if len(names) != len(indices):
        names = []
    return [{"index": index,
             "name": names[i] if names else f"Camera {index}"}
            for i, index in enumerate(indices)]


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
    def __init__(self, model_path="hand_landmarker.task", camera_index=0, use_gpu=True, camera_url=None):
        self.model_path = model_path
        self.camera_index = camera_index
        self.camera_url = camera_url
        self.cap = MJPEGCapture(camera_url) if camera_url else cv2.VideoCapture(self.camera_index)
        self.camera_error = None   # last failed switch, for the web UI
        self._rgb = None      # reusable detection buffer, sized on first frame
        self._last_ts = 0     # monotonic watermark for MediaPipe timestamps
        # A switch is requested from the web server's thread but performed by
        # the gesture loop; this guards the handoff.
        self._switch_lock = threading.Lock()
        self._pending_index = None
        self._use_gpu = use_gpu
        self._confidence_lock = threading.Lock()
        self._confidence = {'detection': 0.8, 'presence': 0.8, 'tracking': 0.8}
        self._pending_confidence = None
        self._confidence_error = None
        self.landmarker = self._init_landmarker(use_gpu, self._confidence)

    def _init_landmarker(self, use_gpu, confidence):
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
            # Require stronger hand evidence to reduce head/background detections.
            min_hand_detection_confidence=confidence["detection"],
            min_hand_presence_confidence=confidence["presence"],
            min_tracking_confidence=confidence["tracking"]
        )
        return HandLandmarker.create_from_options(options)

    def confidence_state(self):
        with self._confidence_lock:
            return {'values': dict(self._pending_confidence or self._confidence),
                    'pending': self._pending_confidence is not None,
                    'error': self._confidence_error}

    def request_confidence(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError("hand_confidence must be a nonempty object")
        validated = {}
        for key, value in values.items():
            if key not in self._confidence or isinstance(value, bool):
                raise ValueError(f"invalid hand confidence: {key}")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"{key} confidence must be between 0 and 1")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} confidence must be between 0 and 1")
            validated[key] = value
        with self._confidence_lock:
            requested = dict(self._pending_confidence or self._confidence)
            requested.update(validated)
            self._pending_confidence = requested if requested != self._confidence else None
            self._confidence_error = None

    def _apply_pending_confidence(self):
        # Only the capture thread replaces the model, between detections.
        with self._confidence_lock:
            requested = self._pending_confidence
            if requested is None:
                return
            try:
                replacement = self._init_landmarker(self._use_gpu, requested)
            except Exception as exc:
                self._confidence_error = f"Could not update hand confidence: {exc}"
                self._pending_confidence = None
                return
            previous = self.landmarker
            self.landmarker = replacement
            self._confidence = requested
            self._pending_confidence = None
            self._confidence_error = None
        previous.close()

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

    def release_camera(self):
        """Let go of the capture device. Called from the gesture loop only.

        The manual tab drives OSC by hand and never looks at the camera, so
        holding the device open there would keep the recording light on and
        keep the camera unavailable to everything else for no reason.
        """
        if self.cap is not None:
            self.cap.release()

    def resume_camera(self):
        """Re-open the device released by release_camera(); gesture loop only.

        Returns whether the camera is open. A failure is recorded for the web
        UI rather than raised: the camera may have been taken by another app
        while the manual tab was up, and the loop keeps running either way.
        """
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.cap.open(self.camera_index):
            self.camera_error = None
            return True
        self.camera_error = f"camera {self.camera_index} would not reopen"
        print("Camera resume failed:", self.camera_error)
        return False

    def request_camera(self, index):
        """Ask for a different camera; the gesture loop performs the swap.

        Releasing the capture from another thread would pull it out from under
        a read in flight, so the request is only recorded here and acted on in
        get_frame_and_landmarks. A second request before the loop gets round to
        the first simply replaces it.
        """
        if self.camera_url:
            raise ValueError("Camera source is set by --camera-url; restart without it to use local cameras")
        index = int(index)
        with self._switch_lock:
            self._pending_index = index
        # A new attempt clears the last one's complaint, so the UI can tell a
        # stale failure from this request failing too.
        self.camera_error = None
        return index

    def _apply_pending_camera(self):
        """Swap in a requested camera. Called from the gesture loop only.

        The new device is opened before the old one is released, so a camera
        that won't open (unplugged, or held by another app) leaves the current
        preview running instead of killing it.
        """
        with self._switch_lock:
            index = self._pending_index
            self._pending_index = None
        if index is None or index == self.camera_index:
            return
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            self.camera_error = f"camera {index} would not open"
            print("Camera switch failed:", self.camera_error)
            return
        self.cap.release()
        self.cap = cap
        self.camera_index = index
        self.camera_error = None
        print("Camera:", index)

    def get_frame_and_landmarks(self, active_area_ratio=1.0):
        self._apply_pending_camera()
        self._apply_pending_confidence()
        if not self.cap.isOpened() and not self._reopen_camera():
            return None, None
        ret, frame = self.cap.read()
        if self.camera_url:
            self.camera_error = self.cap.error
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
