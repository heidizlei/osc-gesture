import time
import cv2
import numpy as np
from pythonosc import udp_client
from .tracker import HandTracker, HandLandmarkDrawer
import gc


class OSCGestureApp:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 camera_index=0,
                 ip="0.0.0.0",
                 port=9001):
        self.interval = (-6, 6)

        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(ip, port)

        # Hand tracking
        self.hand_tracker = HandTracker(
            model_path=model_path,
            camera_index=camera_index,
            use_gpu=False   # GPU = leaks on macOS; CPU recommended
        )

        self.draw_landmarks = False

        self.min_val = 10
        self.max_val = 117
        self.left_val = 63
        self.right_val = 63

        self.last_left = None
        self.last_right = None
        self.change_threshold = 4

        self.running = True

        self.osc_interval = 0.2
        self.inactivity_interval = 1.0
        self.last_osc_time = time.time()
        self.inactivity_message_sent = False

        # Hand presence tracking
        self.hand_present = False
        self.last_hand_present = False

        # Only top part of camera image is active for control
        self.active_area_ratio = 3 / 4


    # ----------------------------
    # OSC sending
    # ----------------------------

    def send_osc_message(self, left_val=None, right_val=None):
        if left_val == -1 and right_val == -1:
            arg1, arg2, arg3, arg4 = 36, 120, -1, -1
        else:
            left_val = left_val if left_val is not None else self.left_val
            right_val = right_val if right_val is not None else self.right_val

            arg1 = left_val + self.interval[0]
            arg2 = left_val + self.interval[1]
            arg3 = right_val + self.interval[0]
            arg4 = right_val + self.interval[1]

        try:
            self.osc_client.send_message("/setOutputRange", [arg1, arg2, arg3, arg4])
            print(f"OSC → /setOutputRange {arg1} {arg2} {arg3} {arg4}")
        except Exception as e:
            print("OSC send error:", e)

    def send_manual_pause(self, pause_flag):
        try:
            self.osc_client.send_message("/setManualPause", pause_flag)
            print(f"OSC → /setManualPause {pause_flag}")
        except Exception as e:
            print("Pause OSC error:", e)


    # ----------------------------
    # Hand position mapping
    # ----------------------------

    def map_hand_x_to_val(self, x_norm):
        if x_norm is None:
            return None
        x_norm = max(0.0, min(1.0, x_norm))
        return int(self.min_val + x_norm * (self.max_val - self.min_val))


    # ----------------------------
    # Events
    # ----------------------------


    # ----------------------------
    # Inactivity
    # ----------------------------

    def handle_inactivity(self):
        now = time.time()
        if now - self.last_osc_time >= self.inactivity_interval and not self.inactivity_message_sent:
            self.send_osc_message(left_val=-1, right_val=-1)
            self.inactivity_message_sent = True


    # ----------------------------
    # Main Loop
    # ----------------------------

    def run(self):
        print("Running OSC Gesture App.")

        frame_counter = 0

        while self.running:
            # ------------------------
            # Camera + Hand Tracking
            # ------------------------
            frame, results = self.hand_tracker.get_frame_and_landmarks()

            if frame is not None:
                annotated = frame  # operate directly on original frame

                h, w, _ = annotated.shape
                inactive_y = int(h * self.active_area_ratio)

                # Transparent overlay for bottom 1/4
                overlay = annotated.copy()
                cv2.rectangle(
                    overlay,
                    (0, inactive_y),
                    (w, h),
                    (100, 100, 255),    # color
                    -1
                )
                alpha = 0.25
                cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

                self.hand_present = bool(results and results.hand_landmarks)

                # Send pause/unpause only when state changes
                if self.hand_present != self.last_hand_present:
                    if self.hand_present:
                        self.send_manual_pause(0)  # resume
                    else:
                        self.send_manual_pause(1)  # pause
                    self.last_hand_present = self.hand_present

                if self.hand_present:
                    positions = []
                    for hand in results.hand_landmarks:
                        xs = [lm.x for lm in hand]
                        ys = [lm.y for lm in hand]
                        positions.append((float(np.mean(xs)), float(np.mean(ys))))

                    # Single-hand → both channels
                    if len(positions) == 1:
                        x, y = positions[0]
                        if y <= self.active_area_ratio:
                            mapped = self.map_hand_x_to_val(x)
                            self.left_val = mapped
                            self.right_val = mapped

                    # Two hands → left/right
                    elif len(positions) >= 2:
                        (x1, y1), (x2, y2) = positions[:2]

                        if y1 <= self.active_area_ratio:
                            self.left_val = self.map_hand_x_to_val(x1)
                        if y2 <= self.active_area_ratio:
                            self.right_val = self.map_hand_x_to_val(x2)

                    # Throttle OSC sends; only fire when change exceeds threshold
                    now = time.time()
                    if now - self.last_osc_time >= self.osc_interval:
                        left_changed = self.last_left is None or abs(self.left_val - self.last_left) > self.change_threshold
                        right_changed = self.last_right is None or abs(self.right_val - self.last_right) > self.change_threshold

                        if left_changed or right_changed:
                            self.last_left = self.left_val
                            self.last_right = self.right_val
                            self.last_osc_time = now
                            self.inactivity_message_sent = False
                            self.send_osc_message()

                # ---- show camera ----
                if self.draw_landmarks:
                    annotated = HandLandmarkDrawer.draw_landmarks(annotated, results)
                cv2.imshow("Hand Camera", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('l'):
                    self.draw_landmarks = not self.draw_landmarks
                    print("Draw landmarks:", self.draw_landmarks)

            if frame_counter % 300 == 0:
                gc.collect()
            frame_counter += 1

        cv2.destroyAllWindows()
        self.hand_tracker.close()
