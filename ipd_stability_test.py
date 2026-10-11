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
# EYE LANDMARKS
# =========================================================

LEFT_EYE_POINTS = [
    33,
    133,
    159,
    145
]

RIGHT_EYE_POINTS = [
    362,
    263,
    386,
    374
]


# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

history = deque(maxlen=5)


# =========================================================
# DATA COLLECTION
# =========================================================

eye_distances = []

total_frames = 0
valid_frames = 0

start_time = None


# =========================================================
# FUNCTION: EYE CENTRE
# =========================================================

def get_eye_center(
    landmarks,
    indices,
    image_w,
    image_h
):

    points_float = []

    for idx in indices:

        landmark = landmarks[idx]

        x = landmark.x * image_w
        y = landmark.y * image_h

        points_float.append(
            (x, y)
        )

    points_float = np.array(
        points_float,
        dtype=np.float64
    )

    center = np.mean(
        points_float,
        axis=0
    )

    center_x = float(center[0])
    center_y = float(center[1])

    return (
        center_x,
        center_y,
        points_float
    )


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(
        rgb
    )

    eye_distance = None


    # =====================================================
    # EYE DISTANCE DETECTION
    # =====================================================

    if results.multi_face_landmarks:

        landmarks = (
            results
            .multi_face_landmarks[0]
            .landmark
        )

        (
            left_cx,
            left_cy,
            left_points
        ) = get_eye_center(
            landmarks,
            LEFT_EYE_POINTS,
            w,
            h
        )

        (
            right_cx,
            right_cy,
            right_points
        ) = get_eye_center(
            landmarks,
            RIGHT_EYE_POINTS,
            w,
            h
        )

        raw_distance = float(
            np.sqrt(
                (right_cx - left_cx) ** 2
                +
                (right_cy - left_cy) ** 2
            )
        )

        history.append(
            raw_distance
        )

        eye_distance = float(
            np.mean(history)
        )


        # -------------------------------------------------
        # VISUALISATION
        # -------------------------------------------------

        for point in left_points:

            px = int(
                round(point[0])
            )

            py = int(
                round(point[1])
            )

            cv2.circle(
                frame,
                (px, py),
                2,
                (255, 0, 0),
                -1
            )


        for point in right_points:

            px = int(
                round(point[0])
            )

            py = int(
                round(point[1])
            )

            cv2.circle(
                frame,
                (px, py),
                2,
                (255, 0, 0),
                -1
            )


        left_center_display = (
            int(round(left_cx)),
            int(round(left_cy))
        )

        right_center_display = (
            int(round(right_cx)),
            int(round(right_cy))
        )


        cv2.circle(
            frame,
            left_center_display,
            5,
            (0, 0, 255),
            -1
        )

        cv2.circle(
            frame,
            right_center_display,
            5,
            (0, 0, 255),
            -1
        )

        cv2.line(
            frame,
            left_center_display,
            right_center_display,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Eye distance: "
            f"{eye_distance:.2f}px",
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
        print("IPD STABILITY TEST STARTED")
        print("Stay still at BASELINE distance.")
        print("Keep approximately the same head pose.")
        print("=====================================")
        print()


    total_frames += 1

    if eye_distance is not None:

        eye_distances.append(
            eye_distance
        )

        valid_frames += 1


    elapsed = (
        time.time()
        -
        start_time
    )

    remaining = max(
        0.0,
        TEST_DURATION - elapsed
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.putText(
        frame,
        "IPD STABILITY TEST - STAY STILL",
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
        "IPD Stability Test",
        frame
    )


    # =====================================================
    # TEST END
    # =====================================================

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
print("IPD STABILITY RESULTS")
print("=====================================")

print(
    f"Total frames: "
    f"{total_frames}"
)

print(
    f"Valid frames: "
    f"{valid_frames}"
)


detection_rate = (
    valid_frames / total_frames
    if total_frames > 0
    else 0.0
)

print(
    f"Detection rate: "
    f"{detection_rate:.3f}"
)


if len(eye_distances) > 0:

    mean_distance = float(
        np.mean(eye_distances)
    )

    std_distance = float(
        np.std(eye_distances)
    )

    stability_score = max(
        0.0,
        1.0
        -
        (
            std_distance
            /
            SIGMA_MAX
        )
    )

    print(
        f"Mean eye distance: "
        f"{mean_distance:.3f} px"
    )

    print(
        f"Std deviation: "
        f"{std_distance:.3f} px"
    )

    print(
        f"Sigma max: "
        f"{SIGMA_MAX:.3f} px"
    )

    print(
        f"IPD stability score: "
        f"{stability_score:.3f}"
    )

else:

    print(
        "No valid eye-distance "
        "measurements were collected."
    )


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()