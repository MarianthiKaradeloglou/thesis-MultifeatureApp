import cv2
import mediapipe as mp
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmarks γύρω από τα μάτια στο MediaPipe Face Mesh
LEFT_EYE_POINTS = [33, 133, 159, 145]
RIGHT_EYE_POINTS = [362, 263, 386, 374]

history = deque(maxlen=5)


def get_center(landmarks, indices, image_w, image_h):
    points = []

    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        points.append((x, y))

    points = np.array(points)
    center = np.mean(points, axis=0)

    return int(center[0]), int(center[1]), points


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eye_distance = None

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_cx, left_cy, left_points = get_center(
            landmarks, LEFT_EYE_POINTS, w, h
        )

        right_cx, right_cy, right_points = get_center(
            landmarks, RIGHT_EYE_POINTS, w, h
        )

        raw_distance = np.sqrt(
            (right_cx - left_cx) ** 2 + (right_cy - left_cy) ** 2
        )

        history.append(raw_distance)
        eye_distance = float(np.mean(history))

        # draw eye landmarks
        for (px, py) in left_points:
            cv2.circle(frame, (px, py), 2, (255, 0, 0), -1)

        for (px, py) in right_points:
            cv2.circle(frame, (px, py), 2, (255, 0, 0), -1)

        # draw eye centers
        cv2.circle(frame, (left_cx, left_cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (right_cx, right_cy), 5, (0, 0, 255), -1)

        # draw distance line
        cv2.line(frame, (left_cx, left_cy), (right_cx, right_cy), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Eye distance: {eye_distance:.2f}px",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        print(f"eye_distance = {eye_distance:.2f}")

    else:
        print("NO FACE / NO EYE DISTANCE DETECTION")

    cv2.imshow("Eye Distance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()