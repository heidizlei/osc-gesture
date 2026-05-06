import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils as mp_drawing

class HandLandmarkDrawer:
    @staticmethod
    def draw_landmarks(image, detection_result):
        annotated_image = np.copy(image)
        if detection_result and hasattr(detection_result, "hand_landmarks") and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
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

    def get_frame_and_landmarks(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
        timestamp = int(time.time() * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp)
        return frame, results

    def close(self):
        self.cap.release()
        try:
            self.landmarker.close()
        except Exception:
            pass
