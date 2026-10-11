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
# TEST SETTINGS
# =========================================================

BASE_PREP_DURATION = 3
BASE_RECORD_DURATION = 5

NEAR_PREP_DURATION = 4
NEAR_RECORD_DURATION = 5

FAR_PREP_DURATION = 4
FAR_RECORD_DURATION = 5

SENSITIVITY_REFERENCE = 0.20


# =========================================================
# DATA STORAGE
# =========================================================

base_values = []
near_values = []
far_values = []

base_total_frames = 0
near_total_frames = 0
far_total_frames = 0

base_valid_frames = 0
near_valid_frames = 0
far_valid_frames = 0

base_two_eye_frames = 0
near_two_eye_frames = 0
far_two_eye_frames = 0

base_single_eye_frames = 0
near_single_eye_frames = 0
far_single_eye_frames = 0


# =========================================================
# AUDIO SIGNALS
# =========================================================

def play_system_sound():
    """
    Plays a predefined Windows system sound.
    """

    winsound.PlaySound(
        "SystemExclamation",
        winsound.SND_ALIAS
    )


def prep_beep():
    """
    PREP:
    One sound means that the preparation phase has started.
    The participant may change position during this phase.
    """

    play_system_sound()


def record_beep():
    """
    RECORD:
    Two rapid sounds indicate the beginning of recording.
    The participant must remain still.
    """

    play_system_sound()

    time.sleep(0.10)

    play_system_sound()


def end_beep():
    """
    END:
    Three rapid sounds indicate the end of the experiment.
    """

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
    duration
):

    prep_beep()

    print(f"\n{name}")
    print("MOVE / PREPARE NOW")

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
            0.9,
            (0, 165, 255),
            2
        )

        cv2.putText(
            frame,
            f"PREP - {remaining:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2
        )

        cv2.imshow(
            "Pupil Sensitivity Test",
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
    print("STAY STILL - RECORDING")

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
            0.9,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"RECORD - {remaining:.1f}s",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
            "Pupil Sensitivity Test",
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
print("PUPIL SENSITIVITY TEST")
print("===================================")

print(
    "ONE SOUND = PREP / MOVE"
)

print(
    "TWO RAPID SOUNDS = RECORD / STAY"
)

print(
    "THREE RAPID SOUNDS = END"
)

print()

print(
    "Keep lighting constant."
)

print(
    "Do not use the screen to identify phases."
)

print("===================================")


# =========================================================
# BASELINE PREP
# =========================================================

if not run_prep_phase(
    "BASELINE PREP",
    BASE_PREP_DURATION
):

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# BASELINE RECORD
# =========================================================

base_result = run_record_phase(
    "BASELINE RECORD",
    BASE_RECORD_DURATION
)

if base_result is None:

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# NEAR PREP
# =========================================================

if not run_prep_phase(
    "NEAR PREP",
    NEAR_PREP_DURATION
):

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# NEAR RECORD
# =========================================================

near_result = run_record_phase(
    "NEAR RECORD",
    NEAR_RECORD_DURATION
)

if near_result is None:

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# FAR PREP
# =========================================================

if not run_prep_phase(
    "FAR PREP",
    FAR_PREP_DURATION
):

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# FAR RECORD
# =========================================================

far_result = run_record_phase(
    "FAR RECORD",
    FAR_RECORD_DURATION
)

if far_result is None:

    cap.release()
    cv2.destroyAllWindows()

    raise SystemExit


# =========================================================
# TEST FINISHED
# =========================================================

end_beep()


# =========================================================
# EXTRACT RESULTS
# =========================================================

base_values = base_result["values"]
near_values = near_result["values"]
far_values = far_result["values"]

base_total_frames = base_result["total_frames"]
near_total_frames = near_result["total_frames"]
far_total_frames = far_result["total_frames"]

base_valid_frames = base_result["valid_frames"]
near_valid_frames = near_result["valid_frames"]
far_valid_frames = far_result["valid_frames"]

base_two_eye_frames = base_result["two_eye_frames"]
near_two_eye_frames = near_result["two_eye_frames"]
far_two_eye_frames = far_result["two_eye_frames"]

base_single_eye_frames = base_result["single_eye_frames"]
near_single_eye_frames = near_result["single_eye_frames"]
far_single_eye_frames = far_result["single_eye_frames"]


# =========================================================
# RESULTS
# =========================================================

print("\n===================================")
print("PUPIL SENSITIVITY TEST RESULTS")
print("===================================")


if (
    len(base_values) > 0
    and len(near_values) > 0
    and len(far_values) > 0
):

    d_base = float(
        np.mean(base_values)
    )

    d_near = float(
        np.mean(near_values)
    )

    d_far = float(
        np.mean(far_values)
    )


    std_base = float(
        np.std(base_values)
    )

    std_near = float(
        np.std(near_values)
    )

    std_far = float(
        np.std(far_values)
    )


    # =====================================================
    # RELATIVE CHANGE
    # =====================================================

    s_near = (
        abs(d_near - d_base)
        / d_base
    )

    s_far = (
        abs(d_far - d_base)
        / d_base
    )


    # =====================================================
    # RAW SENSITIVITY
    # =====================================================

    sensitivity_raw = (
        s_near + s_far
    ) / 2.0


    # =====================================================
    # NORMALIZED SENSITIVITY SCORE
    # =====================================================

    sensitivity_score = min(
        1.0,
        sensitivity_raw
        / SENSITIVITY_REFERENCE
    )


    # =====================================================
    # DETECTION RATES
    # =====================================================

    base_detection = (
        base_valid_frames
        / base_total_frames
        if base_total_frames > 0
        else 0.0
    )

    near_detection = (
        near_valid_frames
        / near_total_frames
        if near_total_frames > 0
        else 0.0
    )

    far_detection = (
        far_valid_frames
        / far_total_frames
        if far_total_frames > 0
        else 0.0
    )


    overall_valid = (
        base_valid_frames
        +
        near_valid_frames
        +
        far_valid_frames
    )

    overall_total = (
        base_total_frames
        +
        near_total_frames
        +
        far_total_frames
    )

    overall_detection = (
        overall_valid
        / overall_total
        if overall_total > 0
        else 0.0
    )


    # =====================================================
    # EXPECTED DIRECTION CHECKS
    # =====================================================

    near_ok = (
        d_near > d_base
    )

    far_ok = (
        d_far < d_base
    )


    # =====================================================
    # BASELINE RESULTS
    # =====================================================

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


    # =====================================================
    # NEAR RESULTS
    # =====================================================

    print("\nNEAR:")

    print(
        f"d_near = {d_near:.3f} px"
    )

    print(
        f"std_near = {std_near:.3f} px"
    )

    print(
        f"valid frames = "
        f"{near_valid_frames}/"
        f"{near_total_frames}"
    )

    print(
        f"detection rate = "
        f"{near_detection:.3f}"
    )

    print(
        f"two-eye frames = "
        f"{near_two_eye_frames}"
    )

    print(
        f"single-eye frames = "
        f"{near_single_eye_frames}"
    )


    # =====================================================
    # FAR RESULTS
    # =====================================================

    print("\nFAR:")

    print(
        f"d_far = {d_far:.3f} px"
    )

    print(
        f"std_far = {std_far:.3f} px"
    )

    print(
        f"valid frames = "
        f"{far_valid_frames}/"
        f"{far_total_frames}"
    )

    print(
        f"detection rate = "
        f"{far_detection:.3f}"
    )

    print(
        f"two-eye frames = "
        f"{far_two_eye_frames}"
    )

    print(
        f"single-eye frames = "
        f"{far_single_eye_frames}"
    )


    # =====================================================
    # SENSITIVITY RESULTS
    # =====================================================

    print("\n-----------------------------------")
    print("SENSITIVITY:")
    print("-----------------------------------")

    print(
        f"s_near = {s_near:.3f}"
    )

    print(
        f"s_far = {s_far:.3f}"
    )

    print(
        f"Sensitivity raw = "
        f"{sensitivity_raw:.3f}"
    )

    print(
        f"Sensitivity score = "
        f"{sensitivity_score:.3f}"
    )

    print(
        f"Overall detection rate = "
        f"{overall_detection:.3f}"
    )


    # =====================================================
    # DIRECTION CHECKS
    # =====================================================

    if near_ok:

        print(
            "Near condition: OK "
            "(diameter increased)"
        )

    else:

        print(
            "Near condition: WARNING "
            "(diameter did not increase)"
        )


    if far_ok:

        print(
            "Far condition: OK "
            "(diameter decreased)"
        )

    else:

        print(
            "Far condition: WARNING "
            "(diameter did not decrease)"
        )


    # =====================================================
    # FORMULA CHECK
    # =====================================================

    print("\n-----------------------------------")
    print("FORMULA CHECK:")
    print("-----------------------------------")


    print(
        "s_near = "
        "|d_near - d_base| / d_base"
    )

    print(
        f"       = "
        f"|{d_near:.3f} - "
        f"{d_base:.3f}| / "
        f"{d_base:.3f}"
    )

    print(
        f"       = {s_near:.3f}"
    )


    print()


    print(
        "s_far = "
        "|d_far - d_base| / d_base"
    )

    print(
        f"      = "
        f"|{d_far:.3f} - "
        f"{d_base:.3f}| / "
        f"{d_base:.3f}"
    )

    print(
        f"      = {s_far:.3f}"
    )


    print()


    print(
        "Sensitivity_raw = "
        "(s_near + s_far) / 2"
    )

    print(
        f"                = "
        f"({s_near:.3f} + "
        f"{s_far:.3f}) / 2"
    )

    print(
        f"                = "
        f"{sensitivity_raw:.3f}"
    )


    print()


    print(
        "Sensitivity_score = "
        "min(1, Sensitivity_raw / 0.20)"
    )

    print(
        f"                  = "
        f"min(1, "
        f"{sensitivity_raw:.3f} / "
        f"{SENSITIVITY_REFERENCE:.2f})"
    )

    print(
        f"                  = "
        f"{sensitivity_score:.3f}"
    )


else:

    print(
        "ERROR: At least one condition "
        "did not contain valid pupil data."
    )


print("\n===================================")
print("TEST COMPLETE")
print("===================================\n")


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()