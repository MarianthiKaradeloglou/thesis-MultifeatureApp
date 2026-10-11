import cv2
import mediapipe as mp
import numpy as np
from collections import deque


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


# =========================================================
# MEDIAPIPE FACE MESH
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# NOSE LANDMARKS
# =========================================================

LEFT_NOSE = 98
RIGHT_NOSE = 327


# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

history = deque(maxlen=5)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    nose_width = None

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        left = landmarks[LEFT_NOSE]
        right = landmarks[RIGHT_NOSE]

        # Floating-point pixel coordinates are used for the measurement.
        x1_float = left.x * w
        y1_float = left.y * h

        x2_float = right.x * w
        y2_float = right.y * h

        nose_width_raw = float(
            np.sqrt(
                (x2_float - x1_float) ** 2
                +
                (y2_float - y1_float) ** 2
            )
        )

        history.append(nose_width_raw)

        nose_width = float(np.mean(history))

        # Integer coordinates are used only for visualisation.
        x1 = int(round(x1_float))
        y1 = int(round(y1_float))

        x2 = int(round(x2_float))
        y2 = int(round(y2_float))

        cv2.circle(
            frame,
            (x1, y1),
            4,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            (x2, y2),
            4,
            (0, 0, 255),
            -1
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Nose width: {nose_width:.2f}px",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        print(
            f"raw_nose_width = {nose_width_raw:.2f}px | "
            f"smoothed_nose_width = {nose_width:.2f}px"
        )

    else:

        print("NO FACE / NO NOSE DETECTION")

    cv2.imshow(
        "Nose Width Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()