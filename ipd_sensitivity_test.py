import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# TEST SETTINGS
# =========================================================

PREP_DURATION = 4.0
RECORD_DURATION = 5.0

SENSITIVITY_REFERENCE = 0.20


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
# TEST PHASES
# =========================================================

phases = [
    ("BASELINE PREP", PREP_DURATION, False),
    ("BASELINE RECORD", RECORD_DURATION, True),
    ("NEAR PREP", PREP_DURATION, False),
    ("NEAR RECORD", RECORD_DURATION, True),
    ("FAR PREP", PREP_DURATION, False),
    ("FAR RECORD", RECORD_DURATION, True)
]

phase_index = 0
phase_start_time = None
current_phase_name = phases[0][0]


# =========================================================
# DATA STORAGE
# =========================================================

phase_values = {
    "BASELINE": [],
    "NEAR": [],
    "FAR": []
}

phase_total_frames = {
    "BASELINE": 0,
    "NEAR": 0,
    "FAR": 0
}

phase_valid_frames = {
    "BASELINE": 0,
    "NEAR": 0,
    "FAR": 0
}


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
# PHASE HELPERS
# =========================================================

def get_recording_group(phase_name):

    if phase_name.startswith("BASELINE"):
        return "BASELINE"

    if phase_name.startswith("NEAR"):
        return "NEAR"

    if phase_name.startswith("FAR"):
        return "FAR"

    return None


def start_phase(index):

    global phase_start_time
    global current_phase_name

    phase_name, phase_duration, recording = phases[index]

    current_phase_name = phase_name
    phase_start_time = time.time()

    history.clear()

    print()
    print("=====================================")
    print(phase_name)
    print("=====================================")

    if phase_name == "BASELINE PREP":

        print(
            "Move to the BASELINE position."
        )

    elif phase_name == "BASELINE RECORD":

        print(
            "Stay still at BASELINE."
        )

    elif phase_name == "NEAR PREP":

        print(
            "Move closer to the camera."
        )

    elif phase_name == "NEAR RECORD":

        print(
            "Stay still at the NEAR position."
        )

    elif phase_name == "FAR PREP":

        print(
            "Move farther from the camera."
        )

    elif phase_name == "FAR RECORD":

        print(
            "Stay still at the FAR position."
        )


# =========================================================
# START FIRST PHASE
# =========================================================

start_phase(
    phase_index
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
    # CURRENT PHASE
    # =====================================================

    (
        phase_name,
        phase_duration,
        recording
    ) = phases[phase_index]

    elapsed = (
        time.time()
        -
        phase_start_time
    )

    remaining = max(
        0.0,
        phase_duration - elapsed
    )


    # =====================================================
    # DATA COLLECTION
    # =====================================================

    if recording:

        group = get_recording_group(
            phase_name
        )

        phase_total_frames[group] += 1

        if eye_distance is not None:

            phase_values[group].append(
                eye_distance
            )

            phase_valid_frames[group] += 1


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.putText(
        frame,
        phase_name,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
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
        "IPD Sensitivity Test",
        frame
    )


    # =====================================================
    # PHASE CHANGE
    # =====================================================

    if elapsed >= phase_duration:

        phase_index += 1

        if phase_index >= len(phases):
            break

        start_phase(
            phase_index
        )


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
print("IPD SENSITIVITY RESULTS")
print("=====================================")


means = {}
stds = {}


for group in [
    "BASELINE",
    "NEAR",
    "FAR"
]:

    values = phase_values[group]

    total = phase_total_frames[group]
    valid = phase_valid_frames[group]

    detection = (
        valid / total
        if total > 0
        else 0.0
    )

    print()
    print(group)

    print(
        f"Total frames: "
        f"{total}"
    )

    print(
        f"Valid frames: "
        f"{valid}"
    )

    print(
        f"Detection rate: "
        f"{detection:.3f}"
    )

    if len(values) > 0:

        means[group] = float(
            np.mean(values)
        )

        stds[group] = float(
            np.std(values)
        )

        print(
            f"Mean eye distance: "
            f"{means[group]:.3f} px"
        )

        print(
            f"Std deviation: "
            f"{stds[group]:.3f} px"
        )

    else:

        means[group] = None
        stds[group] = None

        print(
            "No valid measurements."
        )


# =========================================================
# SENSITIVITY CALCULATION
# =========================================================

if (
    means["BASELINE"] is not None
    and
    means["NEAR"] is not None
    and
    means["FAR"] is not None
):

    baseline_mean = means["BASELINE"]
    near_mean = means["NEAR"]
    far_mean = means["FAR"]


    s_near = abs(
        near_mean - baseline_mean
    ) / baseline_mean


    s_far = abs(
        far_mean - baseline_mean
    ) / baseline_mean


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

    print(
        f"Baseline mean: "
        f"{baseline_mean:.3f} px"
    )

    print(
        f"Baseline std: "
        f"{stds['BASELINE']:.3f} px"
    )

    print(
        f"Near mean: "
        f"{near_mean:.3f} px"
    )

    print(
        f"Near std: "
        f"{stds['NEAR']:.3f} px"
    )

    print(
        f"Far mean: "
        f"{far_mean:.3f} px"
    )

    print(
        f"Far std: "
        f"{stds['FAR']:.3f} px"
    )


    print()
    print("RELATIVE CHANGES")

    print(
        f"s_near: "
        f"{s_near:.3f}"
    )

    print(
        f"s_far: "
        f"{s_far:.3f}"
    )

    print(
        f"Sensitivity raw: "
        f"{sensitivity_raw:.3f}"
    )

    print(
        f"Sensitivity reference: "
        f"{SENSITIVITY_REFERENCE:.3f}"
    )

    print(
        f"IPD sensitivity score: "
        f"{sensitivity_score:.3f}"
    )


    # =====================================================
    # DIRECTION CHECK
    # =====================================================

    print()
    print("DIRECTION CHECK")

    if near_mean > baseline_mean:

        print(
            "NEAR: OK "
            "(eye distance increased)"
        )

    else:

        print(
            "NEAR: WARNING "
            "(eye distance did not increase)"
        )


    if far_mean < baseline_mean:

        print(
            "FAR: OK "
            "(eye distance decreased)"
        )

    else:

        print(
            "FAR: WARNING "
            "(eye distance did not decrease)"
        )


# =========================================================
# OVERALL DETECTION
# =========================================================

total_recorded_frames = sum(
    phase_total_frames.values()
)

total_valid_frames = sum(
    phase_valid_frames.values()
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


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()
cap.release()
cv2.destroyAllWindows()