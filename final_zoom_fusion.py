import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# =========================================================
# FINAL MULTI-FEATURE ZOOM FUSION - VERSION 2
# =========================================================
#
# Features:
#   1. Pupil diameter
#   2. Nose width
#   3. Inter-eye distance (IPD proxy)
#
# Improvements over Version 1:
#   - Consensus-based outlier rejection
#   - Dynamic reliability re-weighting
#   - Temporal smoothing before state decision
#   - Hysteresis around baseline
#
# Final numerical output:
#
#   ZoomControl in [-1, 1]
#
#   ZoomControl > 0  -> user moved closer
#   ZoomControl < 0  -> user moved farther
#   ZoomControl ~ 0  -> user remains near baseline
#
# Discrete state:
#
#   BASELINE
#   NEAR
#   FAR
#
# =========================================================


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
# HAAR CASCADES FOR PUPIL PIPELINE
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_eye_tree_eyeglasses.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load frontal-face Haar cascade."
    )

if eye_cascade.empty():
    raise RuntimeError(
        "Could not load eye Haar cascade."
    )


# =========================================================
# FINAL EXPERIMENTAL RELIABILITY VALUES
# =========================================================

RELIABILITY = {
    "pupil": 0.676,
    "nose": 0.913,
    "ipd": 0.877
}


# =========================================================
# EMPIRICAL RESPONSE SCALES
# =========================================================
#
# Derived from the experimental sensitivity tests.
#
# They place the three feature responses on a comparable
# normalized scale before reliability-weighted fusion.
#
# =========================================================

RESPONSE_SCALE = {
    "pupil": 0.0877,
    "nose": 0.6127,
    "ipd": 0.5813
}


# =========================================================
# SYSTEM PARAMETERS
# =========================================================

# Duration of neutral-position calibration.
CALIBRATION_DURATION = 3.0

# Minimum number of valid samples required to establish
# a baseline for a feature.
MIN_BASELINE_SAMPLES = 10

# Maximum magnitude allowed for each normalized feature
# response before fusion.
FEATURE_CLAMP = 1.5

# Maximum permitted distance from the three-feature median
# before a feature is considered a temporary outlier.
CONSENSUS_THRESHOLD = 0.75

# Exponential smoothing coefficient for the fused signal.
FINAL_SMOOTHING_ALPHA = 0.20

# Hysteresis thresholds.
#
# ENTER_THRESHOLD:
# Required magnitude to leave BASELINE.
#
# EXIT_THRESHOLD:
# Required proximity to zero to return to BASELINE.
ENTER_THRESHOLD = 0.08
EXIT_THRESHOLD = 0.03


# =========================================================
# MEDIAPIPE LANDMARK DEFINITIONS
# =========================================================

LEFT_NOSE = 98
RIGHT_NOSE = 327

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
# TEMPORAL FEATURE HISTORIES
# =========================================================

pupil_left_history = deque(maxlen=5)
pupil_right_history = deque(maxlen=5)

nose_history = deque(maxlen=5)
ipd_history = deque(maxlen=5)


# =========================================================
# CALIBRATION STORAGE
# =========================================================

baseline_samples = {
    "pupil": [],
    "nose": [],
    "ipd": []
}

baseline = {
    "pupil": None,
    "nose": None,
    "ipd": None
}


# =========================================================
# FINAL SYSTEM STATE
# =========================================================

smoothed_zoom = 0.0

zoom_state = "BASELINE"

calibration_start = time.time()
calibration_finished = False


# =========================================================
# HELPER: MEDIAPIPE EYE CENTRE
# =========================================================

def get_eye_center(
    landmarks,
    indices,
    image_w,
    image_h
):

    points = []

    for idx in indices:

        lm = landmarks[idx]

        x = lm.x * image_w
        y = lm.y * image_h

        points.append(
            (x, y)
        )

    points = np.array(
        points,
        dtype=np.float64
    )

    center = np.mean(
        points,
        axis=0
    )

    return (
        float(center[0]),
        float(center[1]),
        points
    )


# =========================================================
# HELPER: SINGLE-EYE PUPIL ESTIMATION
# =========================================================

def estimate_pupil_from_eye_roi(
    gray_eye
):

    if gray_eye is None:
        return None

    if gray_eye.size == 0:
        return None


    eye_h, eye_w = gray_eye.shape[:2]


    y1 = int(
        0.20 * eye_h
    )

    y2 = int(
        0.85 * eye_h
    )

    x1 = int(
        0.10 * eye_w
    )

    x2 = int(
        0.90 * eye_w
    )


    cropped = gray_eye[
        y1:y2,
        x1:x2
    ]


    if cropped.size == 0:
        return None


    # -----------------------------------------------------
    # NORMALIZED EYE ROI
    # -----------------------------------------------------

    resized = cv2.resize(
        cropped,
        (120, 80),
        interpolation=cv2.INTER_LINEAR
    )


    # -----------------------------------------------------
    # CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        resized
    )


    # -----------------------------------------------------
    # GAUSSIAN BLUR
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (7, 7),
        0
    )


    # -----------------------------------------------------
    # ADAPTIVE THRESHOLD
    # -----------------------------------------------------

    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )


    # -----------------------------------------------------
    # MORPHOLOGICAL PROCESSING
    # -----------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    opened = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    processed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


    # -----------------------------------------------------
    # CONTOUR EXTRACTION
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        processed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    roi_center = np.array(
        [
            120.0 / 2.0,
            80.0 / 2.0
        ]
    )


    best_score = None
    best_area = None


    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 40:
            continue

        if area > 1200:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h == 0:
            continue


        aspect_ratio = (
            w / float(h)
        )

        if aspect_ratio < 0.5:
            continue

        if aspect_ratio > 2.0:
            continue


        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue


        circularity = (
            4.0
            *
            np.pi
            *
            area
            /
            (perimeter * perimeter)
        )

        if circularity < 0.35:
            continue


        moments = cv2.moments(
            contour
        )


        if moments["m00"] != 0:

            cx = (
                moments["m10"]
                /
                moments["m00"]
            )

            cy = (
                moments["m01"]
                /
                moments["m00"]
            )

        else:

            cx = x + w / 2.0
            cy = y + h / 2.0


        contour_center = np.array(
            [cx, cy]
        )


        distance_from_center = float(
            np.linalg.norm(
                contour_center
                -
                roi_center
            )
        )


        score = (
            2.0 * circularity
            +
            area / 500.0
            -
            distance_from_center / 100.0
        )


        if (
            best_score is None
            or
            score > best_score
        ):

            best_score = score
            best_area = area


    if best_area is None:
        return None


    equivalent_diameter = (
        2.0
        *
        np.sqrt(
            best_area / np.pi
        )
    )


    return float(
        equivalent_diameter
    )


# =========================================================
# PUPIL FEATURE EXTRACTION
# =========================================================

def detect_pupil_feature(
    frame
):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )


    if len(faces) == 0:
        return None


    face = max(
        faces,
        key=lambda rect:
        rect[2] * rect[3]
    )


    fx, fy, fw, fh = face


    # -----------------------------------------------------
    # UPPER FACE ROI
    # -----------------------------------------------------

    upper_h = int(
        0.55 * fh
    )

    upper_face_gray = gray[
        fy:fy + upper_h,
        fx:fx + fw
    ]


    eyes = eye_cascade.detectMultiScale(
        upper_face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )


    if len(eyes) == 0:
        return None


    eyes = sorted(
        eyes,
        key=lambda rect: rect[0]
    )


    if len(eyes) > 2:
        eyes = eyes[:2]


    pupil_values = []


    for index, (
        ex,
        ey,
        ew,
        eh
    ) in enumerate(eyes):

        eye_gray = upper_face_gray[
            ey:ey + eh,
            ex:ex + ew
        ]


        pupil_diameter = (
            estimate_pupil_from_eye_roi(
                eye_gray
            )
        )


        if pupil_diameter is None:
            continue


        if index == 0:

            pupil_left_history.append(
                pupil_diameter
            )

            smoothed = float(
                np.mean(
                    pupil_left_history
                )
            )

        else:

            pupil_right_history.append(
                pupil_diameter
            )

            smoothed = float(
                np.mean(
                    pupil_right_history
                )
            )


        pupil_values.append(
            smoothed
        )


    if len(pupil_values) == 0:
        return None


    if len(pupil_values) == 1:

        return float(
            pupil_values[0]
        )


    return float(
        np.mean(
            pupil_values[:2]
        )
    )


# =========================================================
# NOSE + IPD FEATURE EXTRACTION
# =========================================================

def detect_face_mesh_features(
    frame
):

    h, w, _ = frame.shape


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = face_mesh.process(
        rgb
    )


    if not results.multi_face_landmarks:

        return (
            None,
            None,
            None
        )


    landmarks = (
        results
        .multi_face_landmarks[0]
        .landmark
    )


    # =====================================================
    # NOSE WIDTH
    # =====================================================

    left_nose = landmarks[
        LEFT_NOSE
    ]

    right_nose = landmarks[
        RIGHT_NOSE
    ]


    nose_x1 = (
        left_nose.x * w
    )

    nose_y1 = (
        left_nose.y * h
    )

    nose_x2 = (
        right_nose.x * w
    )

    nose_y2 = (
        right_nose.y * h
    )


    raw_nose_width = float(
        np.sqrt(
            (nose_x2 - nose_x1) ** 2
            +
            (nose_y2 - nose_y1) ** 2
        )
    )


    nose_history.append(
        raw_nose_width
    )


    nose_width = float(
        np.mean(
            nose_history
        )
    )


    # =====================================================
    # IPD PROXY
    # =====================================================

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


    raw_ipd = float(
        np.sqrt(
            (right_cx - left_cx) ** 2
            +
            (right_cy - left_cy) ** 2
        )
    )


    ipd_history.append(
        raw_ipd
    )


    ipd_value = float(
        np.mean(
            ipd_history
        )
    )


    # =====================================================
    # VISUALIZATION DATA
    # =====================================================

    visual_data = {

        "nose_left": (
            int(round(nose_x1)),
            int(round(nose_y1))
        ),

        "nose_right": (
            int(round(nose_x2)),
            int(round(nose_y2))
        ),

        "eye_left": (
            int(round(left_cx)),
            int(round(left_cy))
        ),

        "eye_right": (
            int(round(right_cx)),
            int(round(right_cy))
        )
    }


    return (
        nose_width,
        ipd_value,
        visual_data
    )


# =========================================================
# RELATIVE RESPONSE
# =========================================================

def calculate_relative_response(
    feature_name,
    current_value
):

    base = baseline[
        feature_name
    ]


    if base is None:
        return None

    if current_value is None:
        return None

    if base <= 0:
        return None


    # -----------------------------------------------------
    # PUPIL SIGN ALIGNMENT
    #
    # Experimental pupil measurements repeatedly showed
    # the opposite direction from the geometric features.
    #
    # Therefore:
    #
    # Positive = NEAR
    # Negative = FAR
    #
    # for all three features.
    # -----------------------------------------------------

    if feature_name == "pupil":

        relative = (
            base
            -
            current_value
        ) / base

    else:

        relative = (
            current_value
            -
            base
        ) / base


    return float(
        relative
    )


# =========================================================
# RESPONSE NORMALIZATION
# =========================================================

def normalize_feature_response(
    feature_name,
    relative_response
):

    if relative_response is None:
        return None


    scale = RESPONSE_SCALE[
        feature_name
    ]


    if scale <= 0:
        return None


    normalized = (
        relative_response
        /
        scale
    )


    normalized = float(
        np.clip(
            normalized,
            -FEATURE_CLAMP,
            FEATURE_CLAMP
        )
    )


    return normalized


# =========================================================
# CONSENSUS-BASED OUTLIER DETECTION
# =========================================================

def apply_consensus_filter(
    normalized_features
):

    available = {
        name: value
        for name, value
        in normalized_features.items()
        if value is not None
    }


    # -----------------------------------------------------
    # Consensus rejection is only possible when all
    # three independent feature estimates are available.
    #
    # With only two features, disagreement alone cannot
    # determine which feature is the outlier.
    # -----------------------------------------------------

    if len(available) != 3:

        return (
            available,
            [],
            None
        )


    values = list(
        available.values()
    )


    median_value = float(
        np.median(
            values
        )
    )


    deviations = {
        name: abs(
            value
            -
            median_value
        )
        for name, value
        in available.items()
    }


    outlier_candidates = [
        name
        for name, deviation
        in deviations.items()
        if deviation
        >
        CONSENSUS_THRESHOLD
    ]


    # -----------------------------------------------------
    # IMPORTANT SAFETY RULE
    #
    # Only ONE feature may be rejected by this mechanism.
    #
    # If two or more features exceed the threshold,
    # consensus is ambiguous. In that case no feature is
    # removed and the normal reliability-weighted fusion
    # is retained.
    # -----------------------------------------------------

    if len(outlier_candidates) == 1:

        outlier = (
            outlier_candidates[0]
        )

        filtered = {
            name: value
            for name, value
            in available.items()
            if name != outlier
        }

        return (
            filtered,
            [outlier],
            median_value
        )


    return (
        available,
        [],
        median_value
    )


# =========================================================
# RELIABILITY-WEIGHTED FUSION
# =========================================================

def fuse_features(
    filtered_features
):

    if len(filtered_features) == 0:

        return (
            None,
            {}
        )


    reliability_sum = sum(
        RELIABILITY[name]
        for name
        in filtered_features
    )


    if reliability_sum <= 0:

        return (
            None,
            {}
        )


    dynamic_weights = {}

    fused_value = 0.0


    for (
        name,
        value
    ) in filtered_features.items():

        weight = (
            RELIABILITY[name]
            /
            reliability_sum
        )


        dynamic_weights[
            name
        ] = weight


        fused_value += (
            weight
            *
            value
        )


    return (
        float(fused_value),
        dynamic_weights
    )


# =========================================================
# HYSTERESIS STATE UPDATE
# =========================================================

def update_zoom_state(
    current_state,
    zoom_value
):

    # -----------------------------------------------------
    # CURRENT STATE: BASELINE
    # -----------------------------------------------------

    if current_state == "BASELINE":

        if zoom_value >= ENTER_THRESHOLD:
            return "NEAR"

        if zoom_value <= -ENTER_THRESHOLD:
            return "FAR"

        return "BASELINE"


    # -----------------------------------------------------
    # CURRENT STATE: NEAR
    # -----------------------------------------------------

    if current_state == "NEAR":

        # Return to baseline only after entering the
        # smaller exit region around zero.
        if abs(zoom_value) <= EXIT_THRESHOLD:
            return "BASELINE"

        # A sufficiently strong direct reversal is allowed.
        if zoom_value <= -ENTER_THRESHOLD:
            return "FAR"

        return "NEAR"


    # -----------------------------------------------------
    # CURRENT STATE: FAR
    # -----------------------------------------------------

    if current_state == "FAR":

        if abs(zoom_value) <= EXIT_THRESHOLD:
            return "BASELINE"

        if zoom_value >= ENTER_THRESHOLD:
            return "NEAR"

        return "FAR"


    return "BASELINE"


# =========================================================
# RESET CALIBRATION
# =========================================================

def reset_calibration():

    global baseline_samples
    global baseline
    global smoothed_zoom
    global zoom_state
    global calibration_start
    global calibration_finished


    baseline_samples = {
        "pupil": [],
        "nose": [],
        "ipd": []
    }


    baseline = {
        "pupil": None,
        "nose": None,
        "ipd": None
    }


    pupil_left_history.clear()
    pupil_right_history.clear()

    nose_history.clear()
    ipd_history.clear()


    smoothed_zoom = 0.0

    zoom_state = "BASELINE"

    calibration_start = (
        time.time()
    )

    calibration_finished = False


# =========================================================
# START
# =========================================================

print()

print(
    "=================================================="
)

print(
    "FINAL MULTI-FEATURE ZOOM FUSION - VERSION 2"
)

print(
    "=================================================="
)

print()

print(
    "Calibration started."
)

print(
    "Stay at a neutral BASELINE position "
    "and keep your face approximately frontal."
)

print()


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


    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    pupil_value = detect_pupil_feature(
        frame
    )


    (
        nose_value,
        ipd_value,
        visual_data
    ) = detect_face_mesh_features(
        frame
    )


    current_features = {

        "pupil": pupil_value,

        "nose": nose_value,

        "ipd": ipd_value
    }


    # =====================================================
    # CALIBRATION PHASE
    # =====================================================

    if not calibration_finished:

        calibration_elapsed = (
            time.time()
            -
            calibration_start
        )


        for (
            feature_name,
            feature_value
        ) in current_features.items():

            if feature_value is not None:

                baseline_samples[
                    feature_name
                ].append(
                    feature_value
                )


        remaining = max(
            0.0,
            CALIBRATION_DURATION
            -
            calibration_elapsed
        )


        cv2.putText(
            frame,
            "CALIBRATION - STAY AT BASELINE",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Time remaining: {remaining:.1f}s",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2
        )


        if (
            calibration_elapsed
            >=
            CALIBRATION_DURATION
        ):

            for feature_name in [
                "pupil",
                "nose",
                "ipd"
            ]:

                samples = (
                    baseline_samples[
                        feature_name
                    ]
                )


                if (
                    len(samples)
                    >=
                    MIN_BASELINE_SAMPLES
                ):

                    baseline[
                        feature_name
                    ] = float(
                        np.mean(
                            samples
                        )
                    )

                else:

                    baseline[
                        feature_name
                    ] = None


            calibration_finished = True


            print()

            print(
                "CALIBRATION FINISHED"
            )

            print(
                "--------------------------------------------------"
            )


            for feature_name in [
                "pupil",
                "nose",
                "ipd"
            ]:

                base_value = (
                    baseline[
                        feature_name
                    ]
                )


                sample_count = len(
                    baseline_samples[
                        feature_name
                    ]
                )


                if base_value is not None:

                    print(
                        f"{feature_name.upper():6s} "
                        f"baseline = "
                        f"{base_value:.3f} px "
                        f"({sample_count} samples)"
                    )

                else:

                    print(
                        f"{feature_name.upper():6s} "
                        f"baseline unavailable "
                        f"({sample_count} samples)"
                    )


            print(
                "--------------------------------------------------"
            )

            print()


    # =====================================================
    # LIVE FUSION
    # =====================================================

    else:

        relative = {}
        normalized = {}


        # -------------------------------------------------
        # RELATIVE + NORMALIZED FEATURE RESPONSES
        # -------------------------------------------------

        for feature_name in [
            "pupil",
            "nose",
            "ipd"
        ]:

            relative[
                feature_name
            ] = calculate_relative_response(
                feature_name,
                current_features[
                    feature_name
                ]
            )


            normalized[
                feature_name
            ] = normalize_feature_response(
                feature_name,
                relative[
                    feature_name
                ]
            )


        # -------------------------------------------------
        # CONSENSUS FILTER
        # -------------------------------------------------

        (
            filtered_features,
            outliers,
            consensus_median
        ) = apply_consensus_filter(
            normalized
        )


        # -------------------------------------------------
        # RELIABILITY-WEIGHTED FUSION
        # -------------------------------------------------

        (
            fused_raw,
            dynamic_weights
        ) = fuse_features(
            filtered_features
        )


        # -------------------------------------------------
        # TEMPORAL SMOOTHING
        # -------------------------------------------------

        if fused_raw is not None:

            smoothed_zoom = (
                FINAL_SMOOTHING_ALPHA
                *
                fused_raw
                +
                (
                    1.0
                    -
                    FINAL_SMOOTHING_ALPHA
                )
                *
                smoothed_zoom
            )


            smoothed_zoom = float(
                np.clip(
                    smoothed_zoom,
                    -1.0,
                    1.0
                )
            )


        # -------------------------------------------------
        # HYSTERESIS STATE DECISION
        # -------------------------------------------------

        zoom_state = update_zoom_state(
            zoom_state,
            smoothed_zoom
        )


        # -------------------------------------------------
        # FINAL NUMERICAL OUTPUT
        #
        # The continuous signal is retained even while the
        # discrete state may be BASELINE.
        # -------------------------------------------------

        zoom_control = float(
            smoothed_zoom
        )


        # =================================================
        # DISPLAY FEATURE INFORMATION
        # =================================================

        y = 40


        for feature_name in [
            "pupil",
            "nose",
            "ipd"
        ]:

            current_value = (
                current_features[
                    feature_name
                ]
            )


            norm_value = (
                normalized[
                    feature_name
                ]
            )


            if current_value is None:

                text = (
                    f"{feature_name.upper()}: "
                    f"NO DETECTION"
                )


            elif norm_value is None:

                text = (
                    f"{feature_name.upper()}: "
                    f"{current_value:.2f}px "
                    f"| no baseline"
                )


            elif feature_name in outliers:

                text = (
                    f"{feature_name.upper()}: "
                    f"{current_value:.2f}px "
                    f"| z={norm_value:+.3f} "
                    f"| OUTLIER"
                )


            else:

                weight = (
                    dynamic_weights.get(
                        feature_name,
                        0.0
                    )
                )


                text = (
                    f"{feature_name.upper()}: "
                    f"{current_value:.2f}px "
                    f"| z={norm_value:+.3f} "
                    f"| w={weight:.3f}"
                )


            cv2.putText(
                frame,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2
            )


            y += 30


        # =================================================
        # DISPLAY CONSENSUS INFORMATION
        # =================================================

        if consensus_median is not None:

            consensus_text = (
                f"Consensus median: "
                f"{consensus_median:+.3f}"
            )


            cv2.putText(
                frame,
                consensus_text,
                (30, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (200, 200, 200),
                1
            )


            y += 25


        # =================================================
        # DISPLAY FINAL ZOOM CONTROL
        # =================================================

        cv2.putText(
            frame,
            f"ZoomControl: "
            f"{zoom_control:+.3f}",
            (30, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (0, 255, 255),
            2
        )


        # =================================================
        # DISPLAY FINAL STATE
        # =================================================

        if zoom_state == "NEAR":

            state_text = (
                "NEAR / ZOOM IN"
            )

        elif zoom_state == "FAR":

            state_text = (
                "FAR / ZOOM OUT"
            )

        else:

            state_text = (
                "BASELINE / STABLE"
            )


        cv2.putText(
            frame,
            state_text,
            (30, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )


        # =================================================
        # DISPLAY OUTLIER STATUS
        # =================================================

        if len(outliers) > 0:

            outlier_text = (
                "Rejected: "
                +
                ", ".join(
                    name.upper()
                    for name
                    in outliers
                )
            )


            cv2.putText(
                frame,
                outlier_text,
                (30, y + 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 0, 255),
                2
            )


        # =================================================
        # CONSOLE OUTPUT
        # =================================================

        weight_pupil = (
            dynamic_weights.get(
                "pupil",
                0.0
            )
        )

        weight_nose = (
            dynamic_weights.get(
                "nose",
                0.0
            )
        )

        weight_ipd = (
            dynamic_weights.get(
                "ipd",
                0.0
            )
        )


        print(
            f"ZoomControl={zoom_control:+.3f}"
            f" | State={zoom_state}"
            f" | P={normalized['pupil']}"
            f" | N={normalized['nose']}"
            f" | I={normalized['ipd']}"
            f" | wP={weight_pupil:.3f}"
            f" | wN={weight_nose:.3f}"
            f" | wI={weight_ipd:.3f}"
            f" | Outliers={outliers}"
        )


    # =====================================================
    # DRAW MEDIAPIPE FEATURE GEOMETRY
    # =====================================================

    if visual_data is not None:

        # -------------------------------------------------
        # NOSE
        # -------------------------------------------------

        cv2.line(
            frame,
            visual_data["nose_left"],
            visual_data["nose_right"],
            (0, 255, 0),
            2
        )


        cv2.circle(
            frame,
            visual_data["nose_left"],
            4,
            (0, 0, 255),
            -1
        )


        cv2.circle(
            frame,
            visual_data["nose_right"],
            4,
            (0, 0, 255),
            -1
        )


        # -------------------------------------------------
        # IPD
        # -------------------------------------------------

        cv2.line(
            frame,
            visual_data["eye_left"],
            visual_data["eye_right"],
            (255, 0, 0),
            2
        )


        cv2.circle(
            frame,
            visual_data["eye_left"],
            5,
            (255, 0, 255),
            -1
        )


        cv2.circle(
            frame,
            visual_data["eye_right"],
            5,
            (255, 0, 255),
            -1
        )


    # =====================================================
    # DISPLAY WINDOW
    # =====================================================

    cv2.imshow(
        "Final Multi-Feature Zoom Fusion V2",
        frame
    )


    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    # =====================================================
    # MANUAL RECALIBRATION
    # =====================================================

    if key == ord("r"):

        reset_calibration()

        print()

        print(
            "MANUAL RECALIBRATION STARTED"
        )

        print(
            "Return to the neutral BASELINE position."
        )

        print()


    # =====================================================
    # EXIT
    # =====================================================

    if key == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

face_mesh.close()

cap.release()

cv2.destroyAllWindows()