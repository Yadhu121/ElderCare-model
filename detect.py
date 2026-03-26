import cv2
import mediapipe as mp
import time
import json
import base64
import threading
import queue
import websocket

ELDER_ID = 25
SECRET_KEY = "eldercare_secure_stream_key_2026"
BACKEND_WS_URL = f"ws://localhost:5259/ws/video?elderId={ELDER_ID}&key={SECRET_KEY}"

CAMERA_INDEX = 0

IDLE_MOVEMENT_THRESHOLD = 5
IDLE_DURATION_THRESHOLD = 10
IDLE_NOTIFICATION_COOLDOWN = 60

class ElderWSClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.send_queue = queue.Queue()

        threading.Thread(target=self._run_forever, daemon=True).start()
        threading.Thread(target=self._sender_worker, daemon=True).start()

    def _run_forever(self):
        while True:
            try:
                print(f"Connecting to {self.url}...")
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever()
            except Exception as e:
                print(f"WS Connection Error: {e}")
            time.sleep(5)

    def on_error(self, ws, error): print(f"WS Error: {error}")
    def on_close(self, ws, status, msg): print("WS Closed")

    def _sender_worker(self):
        while True:
            msg = self.send_queue.get()
            if self.ws and self.ws.sock and self.ws.sock.connected:
                try:
                    self.ws.send(json.dumps(msg))
                except Exception as e:
                    print(f"Failed to send WS message: {e}")
            self.send_queue.task_done()

    def send_event(self, event_type, frame=None):
        msg = {
            "event": event_type,
            "elder_id": ELDER_ID,
            "timestamp": time.time()
        }
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            msg["image"] = base64.b64encode(buffer).decode('utf-8')
        self.send_queue.put(msg)

# REMOVED: send_frame method — Android app handles streaming now

ws_client = ElderWSClient(BACKEND_WS_URL)

cap = cv2.VideoCapture(CAMERA_INDEX)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

prev_hipY = None
prev_time = None

DROP_SPEED_THRESHOLD = 600
MIN_DROP_DISTANCE = 40

idle_start_time = None
last_idle_alert_time = 0

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

while True:
    res, frame = cap.read()
    if not res:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        current_time = time.time()
        h, w, _ = frame.shape
        lm = result.pose_landmarks.landmark

        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        lh_y = int(left_hip.y * h)
        rh_y = int(right_hip.y * h)
        hip_y = (lh_y + rh_y) // 2

        if prev_hipY is not None and prev_time is not None:
            dy = hip_y - prev_hipY
            dt = current_time - prev_time

            if abs(dy) < IDLE_MOVEMENT_THRESHOLD:
                if idle_start_time is None:
                    idle_start_time = current_time
                elif (current_time - idle_start_time) > IDLE_DURATION_THRESHOLD:
                    if (current_time - last_idle_alert_time) > IDLE_NOTIFICATION_COOLDOWN:
                        print(f"Idle alert triggered at {current_time}")
                        ws_client.send_event("IDLE_DETECTED", frame=frame)
                        last_idle_alert_time = current_time
            else:
                idle_start_time = None

            if dt > 0:
                speed = dy / dt
                if speed > DROP_SPEED_THRESHOLD and dy > MIN_DROP_DISTANCE:
                    print("Fall detected! Speed:", int(speed), "Distance:", dy)
                    ws_client.send_event("FALL_DETECTED", frame=frame)

        prev_hipY = hip_y
        prev_time = current_time

        cv2.circle(frame, (w // 2, hip_y), 8, (0, 0, 255), -1)

    # REMOVED: ws_client.send_frame(frame)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()