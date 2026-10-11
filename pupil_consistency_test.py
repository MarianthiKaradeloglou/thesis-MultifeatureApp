import cv2
import numpy as np
import time
import winsound
from collections import deque


# =========================================================
# CAMERA INITIALIZATION
# =========================================================

cap = cv2.VideoCapture(0)


# =========================================================
# HAAR CASCADE INITIALIZATION
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)


# =========================================================
# TEMPORAL SMOOTHING
# =========================================================

history_L = deque(maxlen=5)
history_R = deque(maxlen=5)


# =========================================================
# FIXED EYE ROI SIZE
# =========================================================

FIX_W = 120
FIX_H = 80


# =========================================================
# CONSISTENCY TEST SETTINGS
# =========================================================

PREP_DURATION = 4
RECORD_DURATION = 5

NUMBER_OF_REPETITIONS = 3

EPSILON = 1e-8


# =========================================================
# AUDIO SIGNALS
# =========================================================

def play_system_sound():

    winsound.PlaySound(
        "SystemExclamation",
        winsound.SND_ALIAS
    )


def prep_beep():

    play_system_sound()


def record_beep():

    play_system_sound()

    time.sleep(0.10)

    play_system_sound()


def end_beep():

    play_system_sound()

    time.sleep(0.10)

    play_system_sound()

    time.sleep(0.10)

    play_system_sound()


# =========================================================
# RESET SMOOTHING
# =========================================================

def reset_smoothing():

    history_L.clear()
    history_R.clear()


# =========================================================
# PREPROCESS EYE
# =========================================================

def preprocess_eye(eye_roi):

    h, w = eye_roi.shape

    y1 = int(0.20 * h)
    y2 = int(0.85 * h)

    x1 = int(0.10 * w)
    x2 = int(0.90 * w)

    eye_roi = eye_roi[
        y1:y2,
        x1:x2
    ]

    eye_roi = cv2.resize(
        eye_roi,
        (FIX_W, FIX_H),
        interpolation=cv2.INTER_AREA
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    eye_eq = clahe.apply(
        eye_roi
    )

    eye_blur = cv2.GaussianBlur(
        eye_eq,
        (7, 7),
        0
    )

    thresh = cv2.adaptiveThreshold(
        eye_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return eye_roi, thresh


# =========================================================
# PUPIL DETECTION
# =========================================================

def detect_pupil_size(eye_roi):

    eye_gray, eye_bin = preprocess_eye(
        eye_roi
    )

    contours, _ = cv2.findContours(
        eye_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = -1

    roi_h, roi_w = eye_gray.shape

    cx_roi = roi_w / 2.0
    cy_roi = roi_h / 2.0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 40 or area > 1200:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        aspect = w / float(h)

        if aspect < 0.5 or aspect > 2.0:
            continue

        if circularity < 0.35:
            continue

        M = cv2.moments(
            contour
        )

        if M["m00"] == 0:
            continue

        cx = (
            M["m10"]
            / M["m00"]
        )

        cy = (
            M["m01"]
            / M["m00"]
        )

        dist = np.sqrt(
            (cx - cx_roi) ** 2
            +
            (cy - cy_roi) ** 2
        )

        score = (
            (2.0 * circularity)
            +
            (area / 500.0)
            -
            (dist / 100.0)
        )

        if score > best_score:

            best_score = score
            best = contour

    vis = cv2.cvtColor(
        eye_gray,
        cv2.COLOR_GRAY2BGR
    )

    if best is not None:

        area = cv2.contourArea(
            best
        )

        pupil_diameter = (
            2.0
            * np.sqrt(
                area / np.pi
            )
        )

        M = cv2.moments(
            best
        )

        cx = int(
            M["m10"]
            / M["m00"]
        )

        cy = int(
            M["m01"]
            / M["m00"]
        )

        radius_draw = int(
            pupil_diameter / 2.0
        )

        cv2.circle(
            vis,
            (cx, cy),
            radius_draw,
            (0, 0, 255),
            2
        )

        cv2.circle(
            vis,
            (cx, cy),
            2,
            (0, 255, 0),
            -1
        )

        return (
            pupil_diameter,
            vis,
            eye_bin
        )

    return (
        None,
        vis,
        eye_bin
    )


# =========================================================
# FRAME PROCESSING
# =========================================================

def process_frame():

    ret, frame = cap.read()

    if not ret:

        return (
            None,
            None,
            [],
            "NONE"
        )

    frame = cv2.flip(
        frame,
        1
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    eye_debugs = []
    pupil_sizes = []

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        h_top = int(
            0.55 * h
        )

        roi_gray = gray[
            y:y + h_top,
            x:x + w
        ]

        roi_color = frame[
            y:y + h_top,
            x:x + w
        ]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )

        eyes = sorted(
            eyes,
            key=lambda e: e[0]
        )[:2]

        for i, (ex, ey, ew, eh) in enumerate(eyes):

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

            eye_roi = roi_gray[
                ey:ey + eh,
                ex:ex + ew
            ]

            if eye_roi.size == 0:
                continue

            (
                pupil_diameter,
                eye_vis,
                eye_bin
            ) = detect_pupil_size(
                eye_roi
            )

            if pupil_diameter is not None:

                if i == 0:

                    history_L.append(
                        pupil_diameter
                    )

                    smooth_d = float(
                        np.mean(
                            history_L
                        )
                    )

                else:

                    history_R.append(
                        pupil_diameter
                    )

                    smooth_d = float(
                        np.mean(
                            history_R
                        )
                    )

                pupil_sizes.append(
                    smooth_d
                )

                cv2.putText(
                    roi_color,
                    f"d={smooth_d:.1f}",
                    (
                        ex,
                        max(
                            15,
                            ey - 5
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            eye_debugs.append(
                (
                    f"Eye {i} Gray",
                    eye_vis
                )
            )

            eye_debugs.append(
                (
                    f"Eye {i} Bin",
                    eye_bin
                )
            )

        break

    fused_diameter = None
    detection_type = "NONE"

    if len(pupil_sizes) == 2:

        fused_diameter = float(
            np.mean(
                pupil_sizes
            )
        )

        detection_type = "FUSED"

    elif len(pupil_sizes) == 1:

        fused_diameter = float(
            pupil_sizes[0]
        )

        detection_type = "SINGLE"

    return (
        frame,
        fused_diameter,
        eye_debugs,
        detection_type
    )


# =========================================================
# PREP PHASE
# =========================================================

def run_prep_phase(
    name,
    duration,
    instruction
):

    prep_beep()

    print(
        f"\n{name}"
    )

    print(
        instruction
    )

    start_time = time.time()

    while (
        time.time() - start_time
        < duration
    ):

        (
            frame,
            fused_diameter,
            eye_debugs,
            detection_type
        ) = process_frame()

        if frame is None:
            continue

        remaining = max(
            0,
            duration
            -
            (
                time.time()
                - start_time
            )
        )

        cv2.putText(
            frame,
            name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            f"PREP - {remaining:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 165, 255),
            2
        )

        cv2.imshow(
            "Pupil Consistency Test",
            frame
        )

        for window_name, img in eye_debugs:

            cv2.imshow(
                window_name,
                img
            )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            return False

    return True


# =========================================================
# RECORD PHASE
# =========================================================

def run_record_phase(
    name,
    duration
):

    reset_smoothing()

    record_beep()

    print(
        f"\n{name}"
    )

    print(
        "RECORDING"
    )

    values = []

    total_frames = 0
    valid_frames = 0

    two_eye_frames = 0
    single_eye_frames = 0

    start_time = time.time()

    while (
        time.time() - start_time
        < duration
    ):

        (
            frame,
            fused_diameter,
            eye_debugs,
            detection_type
        ) = process_frame()

        if frame is None:
            continue

        total_frames += 1

        if fused_diameter is not None:

            values.append(
                fused_diameter
            )

            valid_frames += 1

            if detection_type == "FUSED":

                two_eye_frames += 1

            elif detection_type == "SINGLE":

                single_eye_frames += 1

        remaining = max(
            0,
            duration
            -
            (
                time.time()
                - start_time
            )
        )

        cv2.putText(
            frame,
            name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"RECORD - {remaining:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )

        if fused_diameter is not None:

            cv2.putText(
                frame,
                (
                    "Pupil diameter: "
                    f"{fused_diameter:.2f}px"
                ),
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO VALID PUPIL",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Pupil Consistency Test",
            frame
        )

        for window_name, img in eye_debugs:

            cv2.imshow(
                window_name,
                img
            )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            return None

    return {
        "values": values,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "two_eye_frames": two_eye_frames,
        "single_eye_frames": single_eye_frames
    }


# =========================================================
# START TEST
# =========================================================

print("\n===================================")
print("PUPIL CONSISTENCY TEST")
print("===================================")

print(
    "ONE SOUND = PREP / CHANGE POSITION"
)

print(
    "TWO RAPID SOUNDS = RECORD / HOLD POSITION"
)

print(
    "THREE RAPID SOUNDS = END"
)

print()

print(
    "Three Baseline -> Near repetitions"
)

print(
    "will be performed."
)

print()

print(
    "Try to reproduce approximately"
)

print(
    "the same baseline and near positions"
)

print(
    "during all three repetitions."
)

print("===================================")


# =========================================================
# STORAGE FOR THE THREE REPETITIONS
# =========================================================

baseline_results = []
near_results = []


# =========================================================
# THREE REPETITIONS
# =========================================================

for repetition in range(
    1,
    NUMBER_OF_REPETITIONS + 1
):

    # -----------------------------------------------------
    # BASELINE PREP
    # -----------------------------------------------------

    baseline_prep_ok = run_prep_phase(
        f"BASELINE {repetition} PREP",
        PREP_DURATION,
        (
            "Return to the baseline position "
            "and prepare."
        )
    )

    if not baseline_prep_ok:

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


    # -----------------------------------------------------
    # BASELINE RECORD
    # -----------------------------------------------------

    baseline_result = run_record_phase(
        f"BASELINE {repetition} RECORD",
        RECORD_DURATION
    )

    if baseline_result is None:

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit

    baseline_results.append(
        baseline_result
    )


    # -----------------------------------------------------
    # NEAR PREP
    # -----------------------------------------------------

    near_prep_ok = run_prep_phase(
        f"NEAR {repetition} PREP",
        PREP_DURATION,
        (
            "Move to the near position "
            "and prepare."
        )
    )

    if not near_prep_ok:

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit


    # -----------------------------------------------------
    # NEAR RECORD
    # -----------------------------------------------------

    near_result = run_record_phase(
        f"NEAR {repetition} RECORD",
        RECORD_DURATION
    )

    if near_result is None:

        cap.release()
        cv2.destroyAllWindows()

        raise SystemExit

    near_results.append(
        near_result
    )


# =========================================================
# END OF EXPERIMENT
# =========================================================

end_beep()


# =========================================================
# CALCULATE REPETITION RESULTS
# =========================================================

movement_values = []

valid_experiment = True


print("\n===================================")
print("PUPIL CONSISTENCY TEST RESULTS")
print("===================================")


for i in range(
    NUMBER_OF_REPETITIONS
):

    baseline_result = baseline_results[i]
    near_result = near_results[i]

    baseline_values = baseline_result[
        "values"
    ]

    near_values = near_result[
        "values"
    ]


    print(
        f"\n-----------------------------------"
    )

    print(
        f"REPETITION {i + 1}"
    )

    print(
        f"-----------------------------------"
    )


    # -----------------------------------------------------
    # CHECK VALID DATA
    # -----------------------------------------------------

    if (
        len(baseline_values) == 0
        or len(near_values) == 0
    ):

        print(
            "ERROR: Missing valid pupil data."
        )

        valid_experiment = False

        continue


    # -----------------------------------------------------
    # MEAN DIAMETERS
    # -----------------------------------------------------

    d_base = float(
        np.mean(
            baseline_values
        )
    )

    d_near = float(
        np.mean(
            near_values
        )
    )


    # -----------------------------------------------------
    # STANDARD DEVIATIONS
    # -----------------------------------------------------

    std_base = float(
        np.std(
            baseline_values
        )
    )

    std_near = float(
        np.std(
            near_values
        )
    )


    # -----------------------------------------------------
    # DETECTION RATES
    # -----------------------------------------------------

    base_detection = (
        baseline_result["valid_frames"]
        /
        baseline_result["total_frames"]
        if baseline_result["total_frames"] > 0
        else 0.0
    )

    near_detection = (
        near_result["valid_frames"]
        /
        near_result["total_frames"]
        if near_result["total_frames"] > 0
        else 0.0
    )


    # -----------------------------------------------------
    # RELATIVE MOVEMENT
    # -----------------------------------------------------

    movement = (
        abs(
            d_near - d_base
        )
        / d_base
    )

    movement_values.append(
        movement
    )


    # -----------------------------------------------------
    # PRINT REPETITION DATA
    # -----------------------------------------------------

    print(
        f"d_base_{i + 1} = "
        f"{d_base:.3f} px"
    )

    print(
        f"d_near_{i + 1} = "
        f"{d_near:.3f} px"
    )

    print(
        f"std_base_{i + 1} = "
        f"{std_base:.3f} px"
    )

    print(
        f"std_near_{i + 1} = "
        f"{std_near:.3f} px"
    )


    print(
        f"\nBaseline valid frames = "
        f"{baseline_result['valid_frames']}/"
        f"{baseline_result['total_frames']}"
    )

    print(
        f"Baseline detection = "
        f"{base_detection:.3f}"
    )

    print(
        f"Baseline two-eye frames = "
        f"{baseline_result['two_eye_frames']}"
    )

    print(
        f"Baseline single-eye frames = "
        f"{baseline_result['single_eye_frames']}"
    )


    print(
        f"\nNear valid frames = "
        f"{near_result['valid_frames']}/"
        f"{near_result['total_frames']}"
    )

    print(
        f"Near detection = "
        f"{near_detection:.3f}"
    )

    print(
        f"Near two-eye frames = "
        f"{near_result['two_eye_frames']}"
    )

    print(
        f"Near single-eye frames = "
        f"{near_result['single_eye_frames']}"
    )


    print(
        f"\nm_{i + 1} = "
        f"|d_near_{i + 1} - d_base_{i + 1}| "
        f"/ d_base_{i + 1}"
    )

    print(
        f"    = "
        f"|{d_near:.3f} - {d_base:.3f}| "
        f"/ {d_base:.3f}"
    )

    print(
        f"    = {movement:.6f}"
    )


# =========================================================
# FINAL CONSISTENCY CALCULATION
# =========================================================

if (
    valid_experiment
    and len(movement_values)
    == NUMBER_OF_REPETITIONS
):

    movements = np.array(
        movement_values,
        dtype=float
    )


    # -----------------------------------------------------
    # MEAN RELATIVE MOVEMENT
    # -----------------------------------------------------

    mean_movement = float(
        np.mean(
            movements
        )
    )


    # -----------------------------------------------------
    # STANDARD DEVIATION OF RELATIVE MOVEMENT
    # -----------------------------------------------------

    std_movement = float(
        np.std(
            movements
        )
    )


    # -----------------------------------------------------
    # COEFFICIENT OF VARIATION
    # -----------------------------------------------------

    if mean_movement > EPSILON:

        cv_movement = (
            std_movement
            / mean_movement
        )

    else:

        cv_movement = float(
            "inf"
        )


    # -----------------------------------------------------
    # CONTINUOUS CONSISTENCY NORMALIZATION
    # -----------------------------------------------------

    if np.isfinite(
        cv_movement
    ):

        consistency_score = (
            1.0
            /
            (
                1.0
                +
                cv_movement
            )
        )

    else:

        consistency_score = 0.0


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print(
        "\n==================================="
    )

    print(
        "FINAL CONSISTENCY"
    )

    print(
        "==================================="
    )


    print(
        "\nRelative movements:"
    )

    for i, movement in enumerate(
        movement_values,
        start=1
    ):

        print(
            f"m_{i} = "
            f"{movement:.6f}"
        )


    print(
        f"\nMean movement = "
        f"{mean_movement:.6f}"
    )

    print(
        f"Std movement = "
        f"{std_movement:.6f}"
    )


    # =====================================================
    # CV FORMULA CHECK
    # =====================================================

    print(
        "\n-----------------------------------"
    )

    print(
        "COEFFICIENT OF VARIATION"
    )

    print(
        "-----------------------------------"
    )


    print(
        "CV_m = sigma_m / mean_m"
    )

    if np.isfinite(
        cv_movement
    ):

        print(
            f"     = "
            f"{std_movement:.6f} "
            f"/ {mean_movement:.6f}"
        )

        print(
            f"     = "
            f"{cv_movement:.6f}"
        )

    else:

        print(
            "Mean movement is approximately zero."
        )

        print(
            "CV_m is therefore undefined / infinite."
        )


    # =====================================================
    # CONSISTENCY FORMULA CHECK
    # =====================================================

    print(
        "\n-----------------------------------"
    )

    print(
        "CONSISTENCY SCORE"
    )

    print(
        "-----------------------------------"
    )


    print(
        "C_pupil = 1 / (1 + CV_m)"
    )


    if np.isfinite(
        cv_movement
    ):

        print(
            f"        = "
            f"1 / (1 + {cv_movement:.6f})"
        )

        print(
            f"        = "
            f"{consistency_score:.6f}"
        )

    else:

        print(
            "        = 0.000000"
        )


    print(
        "\n-----------------------------------"
    )

    print(
        f"FINAL CONSISTENCY SCORE = "
        f"{consistency_score:.3f}"
    )

    print(
        "-----------------------------------"
    )


else:

    print(
        "\nERROR:"
    )

    print(
        "The consistency score could not "
        "be calculated because at least "
        "one repetition did not contain "
        "valid pupil data."
    )


print(
    "\n==================================="
)

print(
    "TEST COMPLETE"
)

print(
    "===================================\n"
)


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()