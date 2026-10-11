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
# ROBUSTNESS TEST SETTINGS
# =========================================================

PREP_DURATION = 4
RECORD_DURATION = 5

SIGMA_MAX = 3.0
DRIFT_MAX = 0.15


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

    eye_roi = eye_roi[y1:y2, x1:x2]

    eye_roi = cv2.resize(
        eye_roi,
        (FIX_W, FIX_H),
        interpolation=cv2.INTER_AREA
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    eye_eq = clahe.apply(eye_roi)

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

    eye_gray, eye_bin = preprocess_eye(eye_roi)

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

        area = cv2.contourArea(contour)

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

        x, y, w, h = cv2.boundingRect(contour)

        aspect = w / float(h)

        if aspect < 0.5 or aspect > 2.0:
            continue

        if circularity < 0.35:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

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

        area = cv2.contourArea(best)

        pupil_diameter = (
            2.0 * np.sqrt(area / np.pi)
        )

        M = cv2.moments(best)

        cx = int(
            M["m10"] / M["m00"]
        )

        cy = int(
            M["m01"] / M["m00"]
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

        return pupil_diameter, vis, eye_bin

    return None, vis, eye_bin


# =========================================================
# FRAME PROCESSING
# =========================================================

def process_frame():

    ret, frame = cap.read()

    if not ret:
        return None, None, [], "NONE"

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

            pupil_diameter, eye_vis, eye_bin = (
                detect_pupil_size(
                    eye_roi
                )
            )

            if pupil_diameter is not None:

                if i == 0:

                    history_L.append(
                        pupil_diameter
                    )

                    smooth_d = float(
                        np.mean(history_L)
                    )

                else:

                    history_R.append(
                        pupil_diameter
                    )

                    smooth_d = float(
                        np.mean(history_R)
                    )

                pupil_sizes.append(
                    smooth_d
                )

                cv2.putText(
                    roi_color,
                    f"d={smooth_d:.1f}",
                    (ex, max(15, ey - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            eye_debugs.append(
                (f"Eye {i} Gray", eye_vis)
            )

            eye_debugs.append(
                (f"Eye {i} Bin", eye_bin)
            )

        break

    fused_diameter = None
    detection_type = "NONE"

    if len(pupil_sizes) == 2:

        fused_diameter = float(
            np.mean(pupil_sizes)
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

    print(f"\n{name}")
    print(instruction)

    start_time = time.time()

    while (
        time.time() - start_time
        < duration
    ):

        result = process_frame()

        if result[0] is None:
            continue

        (
            frame,
            fused_diameter,
            eye_debugs,
            detection_type
        ) = result

        remaining = max(
            0,
            duration
            -
            (time.time() - start_time)
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
            "Pupil Robustness Test",
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

    print(f"\n{name}")
    print("RECORDING")

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

        result = process_frame()

        if result[0] is None:
            continue

        (
            frame,
            fused_diameter,
            eye_debugs,
            detection_type
        ) = result

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
            (time.time() - start_time)
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
            "Pupil Robustness Test",
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
# SCENARIO METRICS
# =========================================================

def scenario_metrics(
    values,
    valid_frames,
    total_frames,
    d_base
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

    mean_d = float(
        np.mean(values)
    )

    std_d = float(
        np.std(values)
    )

    stability = max(
        0.0,
        1.0 - (
            std_d / SIGMA_MAX
        )
    )

    drift = (
        abs(mean_d - d_base)
        / d_base
        if d_base > 0
        else 1.0
    )

    plausibility = max(
        0.0,
        1.0 - (
            drift / DRIFT_MAX
        )
    )

    score = (
        0.50 * detection
        +
        0.25 * stability
        +
        0.25 * plausibility
    )

    return {
        "mean": mean_d,
        "std": std_d,
        "detection": detection,
        "stability": stability,
        "drift": drift,
        "plausibility": plausibility,
        "score": score
    }


# =========================================================
# START TEST
# =========================================================

print("\n===================================")
print("PUPIL ROBUSTNESS TEST")
print("===================================")

print(
    "ONE SOUND = PREP / CHANGE CONDITION"
)

print(
    "TWO RAPID SOUNDS = RECORD / HOLD CONDITION"
)

print(
    "THREE RAPID SOUNDS = END"
)

print()

print(
    "Keep distance from the camera constant."
)

print("===================================")


# =========================================================
# BASELINE
# =========================================================

if not run_prep_phase(
    "BASELINE PREP",
    PREP_DURATION,
    "Stay frontal, natural position, same distance."
):

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


base_result = run_record_phase(
    "BASELINE RECORD",
    RECORD_DURATION
)

if base_result is None:

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# =========================================================
# TURN
# =========================================================

if not run_prep_phase(
    "TURN PREP",
    PREP_DURATION,
    "Turn your head slightly. Keep the same distance."
):

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


turn_result = run_record_phase(
    "TURN RECORD",
    RECORD_DURATION
)

if turn_result is None:

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# =========================================================
# TILT
# =========================================================

if not run_prep_phase(
    "TILT PREP",
    PREP_DURATION,
    "Tilt your head slightly. Keep the same distance."
):

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


tilt_result = run_record_phase(
    "TILT RECORD",
    RECORD_DURATION
)

if tilt_result is None:

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# =========================================================
# BLINK
# =========================================================

if not run_prep_phase(
    "BLINK PREP",
    PREP_DURATION,
    "Return frontal. Prepare to blink naturally."
):

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


blink_result = run_record_phase(
    "BLINK RECORD",
    RECORD_DURATION
)

if blink_result is None:

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# =========================================================
# LIGHT
# =========================================================

if not run_prep_phase(
    "LIGHT PREP",
    PREP_DURATION,
    "Prepare stronger light on the eye area. Keep same distance."
):

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


light_result = run_record_phase(
    "LIGHT RECORD",
    RECORD_DURATION
)

if light_result is None:

    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit


# =========================================================
# TEST FINISHED
# =========================================================

end_beep()


# =========================================================
# BASELINE METRICS
# =========================================================

base_values = base_result["values"]

base_total_frames = base_result["total_frames"]
base_valid_frames = base_result["valid_frames"]

base_two_eye_frames = base_result["two_eye_frames"]
base_single_eye_frames = base_result["single_eye_frames"]


if len(base_values) == 0:

    print(
        "\nERROR: No valid baseline data collected."
    )

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


d_base = float(
    np.mean(base_values)
)

std_base = float(
    np.std(base_values)
)

base_detection = (
    base_valid_frames / base_total_frames
    if base_total_frames > 0
    else 0.0
)


# =========================================================
# CALCULATE SCENARIO METRICS
# =========================================================

turn_metrics = scenario_metrics(
    turn_result["values"],
    turn_result["valid_frames"],
    turn_result["total_frames"],
    d_base
)

tilt_metrics = scenario_metrics(
    tilt_result["values"],
    tilt_result["valid_frames"],
    tilt_result["total_frames"],
    d_base
)

blink_metrics = scenario_metrics(
    blink_result["values"],
    blink_result["valid_frames"],
    blink_result["total_frames"],
    d_base
)

light_metrics = scenario_metrics(
    light_result["values"],
    light_result["valid_frames"],
    light_result["total_frames"],
    d_base
)


# =========================================================
# FINAL ROBUSTNESS
# =========================================================

robustness_score = (
    turn_metrics["score"]
    +
    tilt_metrics["score"]
    +
    blink_metrics["score"]
    +
    light_metrics["score"]
) / 4.0


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n===================================")
print("PUPIL ROBUSTNESS TEST RESULTS")
print("===================================")


# =========================================================
# BASELINE
# =========================================================

print("\nBASELINE:")

print(
    f"d_base = {d_base:.3f} px"
)

print(
    f"std_base = {std_base:.3f} px"
)

print(
    f"valid frames = "
    f"{base_valid_frames}/"
    f"{base_total_frames}"
)

print(
    f"detection rate = "
    f"{base_detection:.3f}"
)

print(
    f"two-eye frames = "
    f"{base_two_eye_frames}"
)

print(
    f"single-eye frames = "
    f"{base_single_eye_frames}"
)


# =========================================================
# SCENARIO PRINTING FUNCTION
# =========================================================

def print_scenario(
    name,
    metrics,
    result
):

    print(f"\n{name}:")

    if metrics["mean"] is None:

        print(
            "mean = None"
        )

        print(
            "std = None"
        )

    else:

        print(
            f"mean = "
            f"{metrics['mean']:.3f} px"
        )

        print(
            f"std = "
            f"{metrics['std']:.3f} px"
        )

    print(
        f"valid frames = "
        f"{result['valid_frames']}/"
        f"{result['total_frames']}"
    )

    print(
        f"detection = "
        f"{metrics['detection']:.3f}"
    )

    print(
        f"two-eye frames = "
        f"{result['two_eye_frames']}"
    )

    print(
        f"single-eye frames = "
        f"{result['single_eye_frames']}"
    )

    print(
        f"stability = "
        f"{metrics['stability']:.3f}"
    )

    print(
        f"drift = "
        f"{metrics['drift']:.3f}"
    )

    print(
        f"plausibility = "
        f"{metrics['plausibility']:.3f}"
    )

    print(
        f"scenario score = "
        f"{metrics['score']:.3f}"
    )


print_scenario(
    "TURN",
    turn_metrics,
    turn_result
)

print_scenario(
    "TILT",
    tilt_metrics,
    tilt_result
)

print_scenario(
    "BLINK",
    blink_metrics,
    blink_result
)

print_scenario(
    "LIGHT",
    light_metrics,
    light_result
)


# =========================================================
# FINAL RESULT
# =========================================================

print("\n-----------------------------------")
print("FINAL ROBUSTNESS")
print("-----------------------------------")

print(
    "Robustness = "
    "(Turn + Tilt + Blink + Light) / 4"
)

print(
    f"           = "
    f"({turn_metrics['score']:.3f} + "
    f"{tilt_metrics['score']:.3f} + "
    f"{blink_metrics['score']:.3f} + "
    f"{light_metrics['score']:.3f}) / 4"
)

print(
    f"           = "
    f"{robustness_score:.3f}"
)


print("\n===================================")
print("TEST COMPLETE")
print("===================================\n")


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()