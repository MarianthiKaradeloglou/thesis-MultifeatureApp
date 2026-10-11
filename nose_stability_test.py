import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# TEST SETTINGS
# =========================================================

TEST_DURATION = 10.0
SIGMA_MAX = 3.0


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
# DATA COLLECTION
# =========================================================

nose_widths = []

total_frames = 0
valid_frames = 0

start_time = None


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    nose_width = None

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        left = landmarks[LEFT_NOSE]
        right = landmarks[RIGHT_NOSE]

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

    # =====================================================
    # STABILITY TEST
    # =====================================================

    if start_time is None:

        start_time = time.time()

        print()
        print("=====================================")
        print("NOSE STABILITY TEST STARTED")
        print("Stay still at BASELINE distance.")
        print("Keep approximately the same head pose.")
        print("=====================================")
        print()

    total_frames += 1

    if nose_width is not None:

        nose_widths.append(nose_width)
        valid_frames += 1

    elapsed = time.time() - start_time

    remaining = max(
        0.0,
        TEST_DURATION - elapsed
    )

    cv2.putText(
        frame,
        "STABILITY TEST - STAY STILL",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Time remaining: {remaining:.1f}s",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Nose Stability Test",
        frame
    )

    if elapsed >= TEST_DURATION:

        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print()
        print("Test cancelled by user.")

        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


# =========================================================
# RESULTS
# =========================================================

print()
print("=====================================")
print("NOSE STABILITY RESULTS")
print("=====================================")

print(f"Total frames: {total_frames}")
print(f"Valid frames: {valid_frames}")

detection_rate = (
    valid_frames / total_frames
    if total_frames > 0
    else 0.0
)

print(f"Detection rate: {detection_rate:.3f}")

if len(nose_widths) > 0:

    mean_width = float(
        np.mean(nose_widths)
    )

    std_width = float(
        np.std(nose_widths)
    )

    stability_score = max(
        0.0,
        1.0 - (std_width / SIGMA_MAX)
    )

    print(f"Mean nose width: {mean_width:.3f} px")
    print(f"Std deviation: {std_width:.3f} px")
    print(f"Sigma max: {SIGMA_MAX:.3f} px")
    print(f"Nose stability score: {stability_score:.3f}")

else:

    print("No valid nose width measurements were collected.")


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()