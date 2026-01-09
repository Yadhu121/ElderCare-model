import cv2
import mediapipe as mp
import time
import math

cap = cv2.VideoCapture(1)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

prev_hipY = None
prev_time = None

DROP_SPEED_THRESHOLD = 600
MIN_DROP_DISTANCE = 40

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

fall_candidate = False

prev_pos = None
last_movement_time = time.time()

MOVEMENT_THRESHOLD = 10
INACTIVITY_THRESHOLD = 15

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
        
        hip_x = int((left_hip.x + right_hip.x) * 0.5 * w)
        hip_y = int((left_hip.y + right_hip.y) * 0.5 * h)

        current_pos = (hip_x, hip_y)

        if prev_pos is not None:
            dist = math.hypot(
                current_pos[0] - prev_pos[0],
                current_pos[1] - prev_pos[1]
            )

            if dist > MOVEMENT_THRESHOLD:
                last_movement_time = current_time

        prev_pos = current_pos

        inactive_time = current_time - last_movement_time

        if inactive_time > INACTIVITY_THRESHOLD:
            print("Inactive for", int(inactive_time), "seconds")

        if prev_hipY is not None and prev_time is not None:
            dy = hip_y - prev_hipY
            dt = current_time - prev_time

            if dy > 0:
                print("dy is", int(dy))

            if dt > 0:
                speed = dy / dt

                if speed > DROP_SPEED_THRESHOLD and dy > MIN_DROP_DISTANCE:
                    fall_candidate = True
                    print("Fall detected! Speed:", int(speed), "Distance:", dy)

        prev_hipY = hip_y
        prev_time = current_time

        cv2.circle(frame, (w // 2, hip_y), 8, (0, 0, 255), -1)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
