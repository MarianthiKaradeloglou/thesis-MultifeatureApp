import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# TEST SETTINGS
# =========================================================

RECORD_DURATION = 4.0
PREP_DURATION = 3.0


# =========================================================
# TEST PHASES
# =========================================================

PHASE_NAMES = [
    "BASELINE 1 RECORD",
    "NEAR 1 PREP",
    "NEAR 1 RECORD",
    "BASELINE 2 PREP",
    "BASELINE 2 RECORD",
    "NEAR 2 PREP",
    "NEAR 2 RECORD",
    "BASELINE 3 PREP",
    "BASELINE 3 RECORD",
    "NEAR 3 PREP",
    "NEAR 3 RECORD"
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

base1 = []
near1 = []

base2 = []
near2 = []

base3 = []
near3 = []

base1_total = 0
base1_valid = 0

near1_total = 0
near1_valid = 0

base2_total = 0
base2_valid = 0

near2_total = 0
near2_valid = 0

base3_total = 0
base3_valid = 0

near3_total = 0
near3_valid = 0


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

        if current_phase in [0, 4, 8]:

            print(
                "Stay at BASELINE distance."
            )

        elif current_phase in [1, 5, 9]:

            print(
                "Move slightly CLOSER "
                "to the camera."
            )

        elif current_phase in [2, 6, 10]:

            print(
                "Hold the NEAR position."
            )

        elif current_phase in [3, 7]:

            print(
                "Return to the original "
                "BASELINE distance."
            )


    # =====================================================
    # DATA COLLECTION
    # =====================================================

    elapsed = time.time() - phase_start_time

    if current_phase == 0:

        base1_total += 1

        if nose_width is not None:
            base1.append(nose_width)
            base1_valid += 1

    elif current_phase == 2:

        near1_total += 1

        if nose_width is not None:
            near1.append(nose_width)
            near1_valid += 1

    elif current_phase == 4:

        base2_total += 1

        if nose_width is not None:
            base2.append(nose_width)
            base2_valid += 1

    elif current_phase == 6:

        near2_total += 1

        if nose_width is not None:
            near2.append(nose_width)
            near2_valid += 1

    elif current_phase == 8:

        base3_total += 1

        if nose_width is not None:
            base3.append(nose_width)
            base3_valid += 1

    elif current_phase == 10:

        near3_total += 1

        if nose_width is not None:
            near3.append(nose_width)
            near3_valid += 1


    # =====================================================
    # PHASE DURATION
    # =====================================================

    is_record_phase = current_phase in [
        0,
        2,
        4,
        6,
        8,
        10
    ]

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
        "Nose Consistency Test",
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
print("NOSE CONSISTENCY RESULTS")
print("=====================================")


all_sets_valid = (
    len(base1) > 0
    and len(near1) > 0
    and len(base2) > 0
    and len(near2) > 0
    and len(base3) > 0
    and len(near3) > 0
)


if all_sets_valid:

    w_base_1 = float(np.mean(base1))
    w_near_1 = float(np.mean(near1))

    w_base_2 = float(np.mean(base2))
    w_near_2 = float(np.mean(near2))

    w_base_3 = float(np.mean(base3))
    w_near_3 = float(np.mean(near3))


    std_base_1 = float(np.std(base1))
    std_near_1 = float(np.std(near1))

    std_base_2 = float(np.std(base2))
    std_near_2 = float(np.std(near2))

    std_base_3 = float(np.std(base3))
    std_near_3 = float(np.std(near3))


    # =====================================================
    # RELATIVE MOVEMENT FOR EACH REPETITION
    # =====================================================

    m1 = abs(
        w_near_1 - w_base_1
    ) / w_base_1

    m2 = abs(
        w_near_2 - w_base_2
    ) / w_base_2

    m3 = abs(
        w_near_3 - w_base_3
    ) / w_base_3


    movements = [
        m1,
        m2,
        m3
    ]


    mean_movement = float(
        np.mean(movements)
    )

    std_movement = float(
        np.std(movements)
    )


    if mean_movement > 0:

        cv_movement = (
            std_movement
            /
            mean_movement
        )

        consistency_score = (
            1.0
            /
            (1.0 + cv_movement)
        )

    else:

        cv_movement = float("inf")
        consistency_score = 0.0


    # =====================================================
    # DETECTION RATES
    # =====================================================

    base1_detection = (
        base1_valid / base1_total
        if base1_total > 0
        else 0.0
    )

    near1_detection = (
        near1_valid / near1_total
        if near1_total > 0
        else 0.0
    )

    base2_detection = (
        base2_valid / base2_total
        if base2_total > 0
        else 0.0
    )

    near2_detection = (
        near2_valid / near2_total
        if near2_total > 0
        else 0.0
    )

    base3_detection = (
        base3_valid / base3_total
        if base3_total > 0
        else 0.0
    )

    near3_detection = (
        near3_valid / near3_total
        if near3_total > 0
        else 0.0
    )


    # =====================================================
    # REPETITION 1
    # =====================================================

    print()
    print("REPETITION 1")

    print(
        f"Baseline mean: "
        f"{w_base_1:.3f} px"
    )

    print(
        f"Baseline std: "
        f"{std_base_1:.3f} px"
    )

    print(
        f"Baseline frames: "
        f"{base1_valid}/{base1_total}"
    )

    print(
        f"Baseline detection: "
        f"{base1_detection:.3f}"
    )

    print(
        f"Near mean: "
        f"{w_near_1:.3f} px"
    )

    print(
        f"Near std: "
        f"{std_near_1:.3f} px"
    )

    print(
        f"Near frames: "
        f"{near1_valid}/{near1_total}"
    )

    print(
        f"Near detection: "
        f"{near1_detection:.3f}"
    )

    print(
        f"Relative movement m1: "
        f"{m1:.6f}"
    )


    # =====================================================
    # REPETITION 2
    # =====================================================

    print()
    print("REPETITION 2")

    print(
        f"Baseline mean: "
        f"{w_base_2:.3f} px"
    )

    print(
        f"Baseline std: "
        f"{std_base_2:.3f} px"
    )

    print(
        f"Baseline frames: "
        f"{base2_valid}/{base2_total}"
    )

    print(
        f"Baseline detection: "
        f"{base2_detection:.3f}"
    )

    print(
        f"Near mean: "
        f"{w_near_2:.3f} px"
    )

    print(
        f"Near std: "
        f"{std_near_2:.3f} px"
    )

    print(
        f"Near frames: "
        f"{near2_valid}/{near2_total}"
    )

    print(
        f"Near detection: "
        f"{near2_detection:.3f}"
    )

    print(
        f"Relative movement m2: "
        f"{m2:.6f}"
    )


    # =====================================================
    # REPETITION 3
    # =====================================================

    print()
    print("REPETITION 3")

    print(
        f"Baseline mean: "
        f"{w_base_3:.3f} px"
    )

    print(
        f"Baseline std: "
        f"{std_base_3:.3f} px"
    )

    print(
        f"Baseline frames: "
        f"{base3_valid}/{base3_total}"
    )

    print(
        f"Baseline detection: "
        f"{base3_detection:.3f}"
    )

    print(
        f"Near mean: "
        f"{w_near_3:.3f} px"
    )

    print(
        f"Near std: "
        f"{std_near_3:.3f} px"
    )

    print(
        f"Near frames: "
        f"{near3_valid}/{near3_total}"
    )

    print(
        f"Near detection: "
        f"{near3_detection:.3f}"
    )

    print(
        f"Relative movement m3: "
        f"{m3:.6f}"
    )


    # =====================================================
    # CONSISTENCY SCORE
    # =====================================================

    print()
    print("CONSISTENCY METRICS")

    print(
        f"Mean relative movement: "
        f"{mean_movement:.6f}"
    )

    print(
        f"Std relative movement: "
        f"{std_movement:.6f}"
    )

    if np.isfinite(cv_movement):

        print(
            f"Coefficient of variation: "
            f"{cv_movement:.6f}"
        )

    else:

        print(
            "Coefficient of variation: inf"
        )

    print(
        f"Nose consistency score: "
        f"{consistency_score:.3f}"
    )


    # =====================================================
    # OVERALL DETECTION
    # =====================================================

    total_record_frames = (
        base1_total
        +
        near1_total
        +
        base2_total
        +
        near2_total
        +
        base3_total
        +
        near3_total
    )

    total_valid_frames = (
        base1_valid
        +
        near1_valid
        +
        base2_valid
        +
        near2_valid
        +
        base3_valid
        +
        near3_valid
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

    print(
        "Not enough valid data "
        "were collected."
    )


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()