import cv2
import mediapipe as mp
from time import time

previous_avg_shoulder_height = None
last_check_time = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detectPose(frame, pose_model):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    landmarks = []
    h, w, _ = frame.shape

    if not results.pose_landmarks:
        return frame, None

    for lm in results.pose_landmarks.landmark:
        landmarks.append(
            (int(lm.x * w), int(lm.y * h))
        )

    for c in mp_pose.POSE_CONNECTIONS:
        s, e = c
        cv2.line(
            frame,
            landmarks[s],
            landmarks[e],
            (0, 255, 0),
            2
        )

    return frame, landmarks

def detectFall(landmarks, prev_avg):
    left_y = landmarks[11][1]
    right_y = landmarks[12][1]
    avg_y = (left_y + right_y) / 2

    if prev_avg is None:
        return False, avg_y

    drop = avg_y - prev_avg
    DROP_THRESHOLD = 60  ##
    print(f"Prev: {prev_avg:.1f}, Curr: {avg_y:.1f}, Drop: {drop:.1f}")

    if drop > DROP_THRESHOLD:
        return True, prev_avg

    return False, avg_y


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, landmarks = detectPose(frame, pose)

    now = time()
    if landmarks and (now - last_check_time) > 2:
        fall, previous_avg_shoulder_height = detectFall(
            landmarks, previous_avg_shoulder_height
        )

        if fall:
            print("Fall detected!")

        last_check_time = now

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

