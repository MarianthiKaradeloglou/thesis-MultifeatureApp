import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# TEST SETTINGS
# =========================================================

RECORD_DURATION = 5.0
PREP_DURATION = 4.0

SIGMA_MAX = 3.0
DRIFT_MAX = 0.15


# =========================================================
# TEST PHASES
# =========================================================

PHASE_NAMES = [
    "BASELINE RECORD",
    "TURN PREP",
    "TURN RECORD",
    "TILT PREP",
    "TILT RECORD",
    "EXPRESSION PREP",
    "EXPRESSION RECORD",
    "OCCLUSION PREP",
    "OCCLUSION RECORD"
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
turn_widths = []
tilt_widths = []
expression_widths = []
occlusion_widths = []

baseline_total = 0
baseline_valid = 0

turn_total = 0
turn_valid = 0

tilt_total = 0
tilt_valid = 0

expression_total = 0
expression_valid = 0

occlusion_total = 0
occlusion_valid = 0


# =========================================================
# SCENARIO METRICS
# =========================================================

def scenario_metrics(
    values,
    valid_frames,
    total_frames,
    baseline_mean,
    sigma_max,
    drift_max
):

    detection = (
        valid_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    if len(values) == 0:

        return {
            "mean": None,
            "std": None,
            "detection": detection,
            "stability": 0.0,
            "drift": 1.0,
            "plausibility": 0.0,
            "score": 0.5 * detection
        }

    mean_value = float(
        np.mean(values)
    )

    std_value = float(
        np.std(values)
    )

    stability = max(
        0.0,
        1.0 - (std_value / sigma_max)
    )

    if baseline_mean > 0:

        drift = abs(
            mean_value - baseline_mean
        ) / baseline_mean

    else:

        drift = 1.0

    plausibility = max(
        0.0,
        1.0 - (drift / drift_max)
    )

    score = (
        0.50 * detection
        +
        0.25 * stability
        +
        0.25 * plausibility
    )

    return {
        "mean": mean_value,
        "std": std_value,
        "detection": detection,
        "stability": stability,
        "drift": drift,
        "plausibility": plausibility,
        "score": score
    }


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

            print(
                "Stay at BASELINE distance "
                "with a frontal head pose."
            )

        elif current_phase == 1:

            print(
                "Turn your head slightly "
                "to one side."
            )

        elif current_phase == 2:

            print(
                "Hold the turned head position."
            )

        elif current_phase == 3:

            print(
                "Return to frontal position "
                "and tilt your head slightly "
                "towards one shoulder."
            )

        elif current_phase == 4:

            print(
                "Hold the tilted head position."
            )

        elif current_phase == 5:

            print(
                "Return to frontal position "
                "and prepare to make a facial expression."
            )

        elif current_phase == 6:

            print(
                "Hold a natural facial expression "
                "such as a smile."
            )

        elif current_phase == 7:

            print(
                "Return to neutral expression "
                "and prepare to partially occlude the nose."
            )

        elif current_phase == 8:

            print(
                "Partially cover one side of the nose "
                "without covering the entire face."
            )


    # =====================================================
    # DATA COLLECTION
    # =====================================================

    elapsed = time.time() - phase_start_time

    if current_phase == 0:

        baseline_total += 1

        if nose_width is not None:
            baseline_widths.append(nose_width)
            baseline_valid += 1

    elif current_phase == 2:

        turn_total += 1

        if nose_width is not None:
            turn_widths.append(nose_width)
            turn_valid += 1

    elif current_phase == 4:

        tilt_total += 1

        if nose_width is not None:
            tilt_widths.append(nose_width)
            tilt_valid += 1

    elif current_phase == 6:

        expression_total += 1

        if nose_width is not None:
            expression_widths.append(nose_width)
            expression_valid += 1

    elif current_phase == 8:

        occlusion_total += 1

        if nose_width is not None:
            occlusion_widths.append(nose_width)
            occlusion_valid += 1


    # =====================================================
    # PHASE DURATION
    # =====================================================

    is_record_phase = current_phase in [
        0,
        2,
        4,
        6,
        8
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
        "Nose Robustness Test",
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
print("NOSE ROBUSTNESS RESULTS")
print("=====================================")


if len(baseline_widths) == 0:

    print(
        "No valid baseline measurements "
        "were collected."
    )

else:

    baseline_mean = float(
        np.mean(baseline_widths)
    )

    baseline_std = float(
        np.std(baseline_widths)
    )

    baseline_detection = (
        baseline_valid / baseline_total
        if baseline_total > 0
        else 0.0
    )


    print()
    print("BASELINE")
    print(f"Total frames: {baseline_total}")
    print(f"Valid frames: {baseline_valid}")
    print(
        f"Detection rate: "
        f"{baseline_detection:.3f}"
    )
    print(
        f"Mean nose width: "
        f"{baseline_mean:.3f} px"
    )
    print(
        f"Std deviation: "
        f"{baseline_std:.3f} px"
    )


    turn_metrics = scenario_metrics(
        turn_widths,
        turn_valid,
        turn_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    tilt_metrics = scenario_metrics(
        tilt_widths,
        tilt_valid,
        tilt_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    expression_metrics = scenario_metrics(
        expression_widths,
        expression_valid,
        expression_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    occlusion_metrics = scenario_metrics(
        occlusion_widths,
        occlusion_valid,
        occlusion_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )


    metrics_list = [
        (
            "TURN",
            turn_metrics,
            turn_total,
            turn_valid
        ),
        (
            "TILT",
            tilt_metrics,
            tilt_total,
            tilt_valid
        ),
        (
            "EXPRESSION",
            expression_metrics,
            expression_total,
            expression_valid
        ),
        (
            "OCCLUSION",
            occlusion_metrics,
            occlusion_total,
            occlusion_valid
        )
    ]


    for name, metrics, total_frames, valid_frames in metrics_list:

        print()
        print(name)

        if metrics["mean"] is not None:

            print(
                f"Mean nose width: "
                f"{metrics['mean']:.3f} px"
            )

            print(
                f"Std deviation: "
                f"{metrics['std']:.3f} px"
            )

        else:

            print("Mean nose width: None")
            print("Std deviation: None")

        print(
            f"Total frames: "
            f"{total_frames}"
        )

        print(
            f"Valid frames: "
            f"{valid_frames}"
        )

        print(
            f"Detection rate: "
            f"{metrics['detection']:.3f}"
        )

        print(
            f"Stability: "
            f"{metrics['stability']:.3f}"
        )

        print(
            f"Drift: "
            f"{metrics['drift']:.3f}"
        )

        print(
            f"Plausibility: "
            f"{metrics['plausibility']:.3f}"
        )

        print(
            f"Scenario score: "
            f"{metrics['score']:.3f}"
        )


    robustness_score = float(
        np.mean(
            [
                turn_metrics["score"],
                tilt_metrics["score"],
                expression_metrics["score"],
                occlusion_metrics["score"]
            ]
        )
    )


    # =====================================================
    # OVERALL DETECTION
    # =====================================================

    total_record_frames = (
        baseline_total
        +
        turn_total
        +
        tilt_total
        +
        expression_total
        +
        occlusion_total
    )

    total_valid_frames = (
        baseline_valid
        +
        turn_valid
        +
        tilt_valid
        +
        expression_valid
        +
        occlusion_valid
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

    print(
        f"Final nose robustness score: "
        f"{robustness_score:.3f}"
    )


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()