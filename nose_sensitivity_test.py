import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# TEST SETTINGS
# =========================================================

RECORD_DURATION = 5.0
PREP_DURATION = 3.0

SENSITIVITY_REFERENCE = 0.20


# =========================================================
# TEST PHASES
# =========================================================

PHASE_NAMES = [
    "BASELINE RECORD",
    "NEAR PREP",
    "NEAR RECORD",
    "FAR PREP",
    "FAR RECORD"
]

current_phase = 0
phase_start_time = None
phase_initialized = False


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
# DATA STORAGE
# =========================================================

baseline_widths = []
near_widths = []
far_widths = []

baseline_total = 0
baseline_valid = 0

near_total = 0
near_valid = 0

far_total = 0
far_valid = 0


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


    # =====================================================
    # NOSE WIDTH DETECTION
    # =====================================================

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

        nose_width = float(
            np.mean(history)
        )

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
    # PHASE INITIALISATION
    # =====================================================

    if phase_start_time is None:

        phase_start_time = time.time()

    if not phase_initialized:

        phase_initialized = True

        print()
        print("=====================================")
        print(f"CURRENT PHASE: {PHASE_NAMES[current_phase]}")
        print("=====================================")

        if current_phase == 0:
            print("Stay at BASELINE distance.")

        elif current_phase == 1:
            print("Move slightly CLOSER to the camera.")

        elif current_phase == 2:
            print("Hold the NEAR position.")

        elif current_phase == 3:
            print("Move FARTHER from the camera.")

        elif current_phase == 4:
            print("Hold the FAR position.")


    # =====================================================
    # DATA COLLECTION
    # =====================================================

    elapsed = time.time() - phase_start_time

    if current_phase == 0:

        baseline_total += 1

        if nose_width is not None:

            baseline_widths.append(
                nose_width
            )

            baseline_valid += 1


    elif current_phase == 2:

        near_total += 1

        if nose_width is not None:

            near_widths.append(
                nose_width
            )

            near_valid += 1


    elif current_phase == 4:

        far_total += 1

        if nose_width is not None:

            far_widths.append(
                nose_width
            )

            far_valid += 1


    # =====================================================
    # PHASE DURATION
    # =====================================================

    is_record_phase = current_phase in [0, 2, 4]

    if is_record_phase:

        current_duration = RECORD_DURATION

    else:

        current_duration = PREP_DURATION


    remaining = max(
        0.0,
        current_duration - elapsed
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.putText(
        frame,
        PHASE_NAMES[current_phase],
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
        "Nose Sensitivity Test",
        frame
    )


    # =====================================================
    # PHASE TRANSITION
    # =====================================================

    if elapsed >= current_duration:

        current_phase += 1

        phase_start_time = None
        phase_initialized = False

        if current_phase >= len(PHASE_NAMES):

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
print("NOSE SENSITIVITY RESULTS")
print("=====================================")


baseline_detection = (
    baseline_valid / baseline_total
    if baseline_total > 0
    else 0.0
)

near_detection = (
    near_valid / near_total
    if near_total > 0
    else 0.0
)

far_detection = (
    far_valid / far_total
    if far_total > 0
    else 0.0
)


print()
print("BASELINE")
print(f"Total frames: {baseline_total}")
print(f"Valid frames: {baseline_valid}")
print(f"Detection rate: {baseline_detection:.3f}")

print()
print("NEAR")
print(f"Total frames: {near_total}")
print(f"Valid frames: {near_valid}")
print(f"Detection rate: {near_detection:.3f}")

print()
print("FAR")
print(f"Total frames: {far_total}")
print(f"Valid frames: {far_valid}")
print(f"Detection rate: {far_detection:.3f}")


if (
    len(baseline_widths) > 0
    and len(near_widths) > 0
    and len(far_widths) > 0
):

    w_base = float(
        np.mean(baseline_widths)
    )

    w_near = float(
        np.mean(near_widths)
    )

    w_far = float(
        np.mean(far_widths)
    )

    std_base = float(
        np.std(baseline_widths)
    )

    std_near = float(
        np.std(near_widths)
    )

    std_far = float(
        np.std(far_widths)
    )


    s_near = abs(
        w_near - w_base
    ) / w_base

    s_far = abs(
        w_far - w_base
    ) / w_base


    sensitivity_raw = (
        s_near + s_far
    ) / 2.0


    sensitivity_score = min(
        1.0,
        sensitivity_raw
        /
        SENSITIVITY_REFERENCE
    )


    print()
    print("MEASUREMENTS")
    print(f"Baseline mean: {w_base:.3f} px")
    print(f"Baseline std:  {std_base:.3f} px")

    print(f"Near mean:     {w_near:.3f} px")
    print(f"Near std:      {std_near:.3f} px")

    print(f"Far mean:      {w_far:.3f} px")
    print(f"Far std:       {std_far:.3f} px")


    print()
    print("RELATIVE CHANGES")
    print(f"s_near: {s_near:.3f}")
    print(f"s_far:  {s_far:.3f}")

    print(
        f"Sensitivity raw: "
        f"{sensitivity_raw:.3f}"
    )

    print(
        f"Sensitivity reference: "
        f"{SENSITIVITY_REFERENCE:.3f}"
    )

    print(
        f"Nose sensitivity score: "
        f"{sensitivity_score:.3f}"
    )


    # =====================================================
    # DIRECTION CHECK
    # =====================================================

    print()
    print("DIRECTION CHECK")

    if w_near > w_base:

        print(
            "NEAR: OK "
            "(nose width increased)"
        )

    else:

        print(
            "NEAR: WARNING "
            "(nose width did not increase)"
        )


    if w_far < w_base:

        print(
            "FAR: OK "
            "(nose width decreased)"
        )

    else:

        print(
            "FAR: WARNING "
            "(nose width did not decrease)"
        )


    # =====================================================
    # OVERALL DETECTION
    # =====================================================

    total_record_frames = (
        baseline_total
        +
        near_total
        +
        far_total
    )

    total_valid_frames = (
        baseline_valid
        +
        near_valid
        +
        far_valid
    )

    overall_detection = (
        total_valid_frames
        /
        total_record_frames
        if total_record_frames > 0
        else 0.0
    )

    print()
    print("OVERALL")
    print(
        f"Total recorded frames: "
        f"{total_record_frames}"
    )

    print(
        f"Total valid frames: "
        f"{total_valid_frames}"
    )

    print(
        f"Overall detection rate: "
        f"{overall_detection:.3f}"
    )

else:

    print()
    print(
        "Not enough valid measurements "
        "were collected."
    )


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()