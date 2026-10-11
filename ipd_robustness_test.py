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
    "SQUINT PREP",
    "SQUINT RECORD",
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
# DATA STORAGE
# =========================================================

baseline_values = []
turn_values = []
tilt_values = []
squint_values = []
occlusion_values = []

baseline_total = 0
baseline_valid = 0

turn_total = 0
turn_valid = 0

tilt_total = 0
tilt_valid = 0

squint_total = 0
squint_valid = 0

occlusion_total = 0
occlusion_valid = 0


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
    # IPD FEATURE EXTRACTION
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
    # PHASE INITIALISATION
    # =====================================================

    if phase_start_time is None:

        phase_start_time = time.time()

    if not phase_initialized:

        phase_initialized = True

        history.clear()

        print()
        print("=====================================")
        print(
            f"CURRENT PHASE: "
            f"{PHASE_NAMES[current_phase]}"
        )
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
                "and prepare to squint."
            )

        elif current_phase == 6:

            print(
                "Keep both eyes partially narrowed "
                "without closing them completely."
            )

        elif current_phase == 7:

            print(
                "Return to normal eye opening "
                "and prepare to partially cover one eye."
            )

        elif current_phase == 8:

            print(
                "Partially cover one eye "
                "without covering the entire face."
            )


    # =====================================================
    # DATA COLLECTION
    # =====================================================

    elapsed = (
        time.time()
        -
        phase_start_time
    )

    if current_phase == 0:

        baseline_total += 1

        if eye_distance is not None:

            baseline_values.append(
                eye_distance
            )

            baseline_valid += 1


    elif current_phase == 2:

        turn_total += 1

        if eye_distance is not None:

            turn_values.append(
                eye_distance
            )

            turn_valid += 1


    elif current_phase == 4:

        tilt_total += 1

        if eye_distance is not None:

            tilt_values.append(
                eye_distance
            )

            tilt_valid += 1


    elif current_phase == 6:

        squint_total += 1

        if eye_distance is not None:

            squint_values.append(
                eye_distance
            )

            squint_valid += 1


    elif current_phase == 8:

        occlusion_total += 1

        if eye_distance is not None:

            occlusion_values.append(
                eye_distance
            )

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
        f"Time remaining: "
        f"{remaining:.1f}s",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "IPD Robustness Test",
        frame
    )


    # =====================================================
    # PHASE TRANSITION
    # =====================================================

    if elapsed >= current_duration:

        current_phase += 1

        phase_start_time = None
        phase_initialized = False

        if current_phase >= len(
            PHASE_NAMES
        ):
            break


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print()
        print(
            "Test cancelled by user."
        )

        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


# =========================================================
# RESULTS
# =========================================================

print()
print("=====================================")
print("IPD ROBUSTNESS RESULTS")
print("=====================================")


if len(baseline_values) == 0:

    print(
        "No valid baseline measurements "
        "were collected."
    )

else:

    baseline_mean = float(
        np.mean(baseline_values)
    )

    baseline_std = float(
        np.std(baseline_values)
    )

    baseline_detection = (
        baseline_valid / baseline_total
        if baseline_total > 0
        else 0.0
    )


    print()
    print("BASELINE")

    print(
        f"Total frames: "
        f"{baseline_total}"
    )

    print(
        f"Valid frames: "
        f"{baseline_valid}"
    )

    print(
        f"Detection rate: "
        f"{baseline_detection:.3f}"
    )

    print(
        f"Mean eye distance: "
        f"{baseline_mean:.3f} px"
    )

    print(
        f"Std deviation: "
        f"{baseline_std:.3f} px"
    )


    turn_metrics = scenario_metrics(
        turn_values,
        turn_valid,
        turn_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    tilt_metrics = scenario_metrics(
        tilt_values,
        tilt_valid,
        tilt_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    squint_metrics = scenario_metrics(
        squint_values,
        squint_valid,
        squint_total,
        baseline_mean,
        SIGMA_MAX,
        DRIFT_MAX
    )

    occlusion_metrics = scenario_metrics(
        occlusion_values,
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
            "SQUINT",
            squint_metrics,
            squint_total,
            squint_valid
        ),
        (
            "OCCLUSION",
            occlusion_metrics,
            occlusion_total,
            occlusion_valid
        )
    ]


    for (
        name,
        metrics,
        total_frames,
        valid_frames
    ) in metrics_list:

        print()
        print(name)

        if metrics["mean"] is not None:

            print(
                f"Mean eye distance: "
                f"{metrics['mean']:.3f} px"
            )

            print(
                f"Std deviation: "
                f"{metrics['std']:.3f} px"
            )

        else:

            print(
                "Mean eye distance: None"
            )

            print(
                "Std deviation: None"
            )

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


    # =====================================================
    # FINAL ROBUSTNESS
    # =====================================================

    robustness_score = float(
        np.mean(
            [
                turn_metrics["score"],
                tilt_metrics["score"],
                squint_metrics["score"],
                occlusion_metrics["score"]
            ]
        )
    )


    # =====================================================
    # OVERALL DETECTION
    # =====================================================

    total_recorded_frames = (
        baseline_total
        +
        turn_total
        +
        tilt_total
        +
        squint_total
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
        squint_valid
        +
        occlusion_valid
    )

    overall_detection = (
        total_valid_frames
        /
        total_recorded_frames
        if total_recorded_frames > 0
        else 0.0
    )


    print()
    print("OVERALL")

    print(
        f"Total recorded frames: "
        f"{total_recorded_frames}"
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
        f"Final IPD robustness score: "
        f"{robustness_score:.3f}"
    )


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()