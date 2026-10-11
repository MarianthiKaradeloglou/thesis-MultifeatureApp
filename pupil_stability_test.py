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

# Ξεχωριστό ιστορικό διαμέτρου για κάθε μάτι.
# Διατηρούνται οι τελευταίες 5 έγκυρες μετρήσεις.
history_L = deque(maxlen=5)
history_R = deque(maxlen=5)


# =========================================================
# FIXED EYE ROI SIZE
# =========================================================

FIX_W = 120
FIX_H = 80


# =========================================================
# STABILITY TEST SETTINGS
# =========================================================

prep_duration = 3       # χρόνος προετοιμασίας πριν την καταγραφή
test_duration = 10      # διάρκεια πραγματικής καταγραφής σε sec

sigma_max = 3.0         # όριο κανονικοποίησης stability score

diameters = []

total_frames = 0
valid_frames = 0

two_eye_frames = 0
single_eye_frames = 0

test_started = False
test_finished = False

program_start_time = time.time()
test_start_time = None


# =========================================================
# AUDIO SIGNAL
# =========================================================

def beep(freq=1000, duration=250):
    winsound.Beep(freq, duration)


# =========================================================
# EYE ROI PREPROCESSING
# =========================================================

def preprocess_eye(eye_roi):

    # -----------------------------------------------------
    # 1. Crop της περιοχής του ματιού
    # -----------------------------------------------------

    h, w = eye_roi.shape

    y1 = int(0.20 * h)
    y2 = int(0.85 * h)

    x1 = int(0.10 * w)
    x2 = int(0.90 * w)

    eye_roi = eye_roi[y1:y2, x1:x2]


    # -----------------------------------------------------
    # 2. Resize σε σταθερό μέγεθος
    # -----------------------------------------------------

    eye_roi = cv2.resize(
        eye_roi,
        (FIX_W, FIX_H),
        interpolation=cv2.INTER_AREA
    )


    # -----------------------------------------------------
    # 3. Βελτίωση τοπικής αντίθεσης με CLAHE
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    eye_eq = clahe.apply(eye_roi)


    # -----------------------------------------------------
    # 4. Gaussian Blur για μείωση θορύβου
    # -----------------------------------------------------

    eye_blur = cv2.GaussianBlur(
        eye_eq,
        (7, 7),
        0
    )


    # -----------------------------------------------------
    # 5. Adaptive Thresholding
    # -----------------------------------------------------

    thresh = cv2.adaptiveThreshold(
        eye_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )


    # -----------------------------------------------------
    # 6. Morphological filtering
    # -----------------------------------------------------

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
# PUPIL SIZE DETECTION
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


    # -----------------------------------------------------
    # Κέντρο του eye ROI
    # -----------------------------------------------------

    roi_h, roi_w = eye_gray.shape

    cx_roi = roi_w / 2.0
    cy_roi = roi_h / 2.0


    # -----------------------------------------------------
    # Αξιολόγηση όλων των contours
    # -----------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(contour)


        # Λογικά όρια εμβαδού
        if area < 40 or area > 1200:
            continue


        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue


        # -------------------------------------------------
        # Circularity
        # -------------------------------------------------

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )


        # -------------------------------------------------
        # Aspect ratio
        # -------------------------------------------------

        x, y, w, h = cv2.boundingRect(contour)

        aspect = w / float(h)


        if aspect < 0.5 or aspect > 2.0:
            continue


        if circularity < 0.35:
            continue


        # -------------------------------------------------
        # Κέντρο contour μέσω moments
        # -------------------------------------------------

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue


        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]


        # -------------------------------------------------
        # Απόσταση contour από κέντρο ROI
        # -------------------------------------------------

        dist = np.sqrt(
            (cx - cx_roi) ** 2
            +
            (cy - cy_roi) ** 2
        )


        # -------------------------------------------------
        # Συνολικό contour score
        # -------------------------------------------------

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


    # =====================================================
    # VISUALIZATION IMAGE
    # =====================================================

    vis = cv2.cvtColor(
        eye_gray,
        cv2.COLOR_GRAY2BGR
    )


    # =====================================================
    # VALID PUPIL FOUND
    # =====================================================

    if best is not None:

        area = cv2.contourArea(best)


        # -------------------------------------------------
        # Equivalent pupil diameter
        # d = 2 * sqrt(A / pi)
        # -------------------------------------------------

        pupil_diameter = (
            2.0 * np.sqrt(area / np.pi)
        )


        # -------------------------------------------------
        # Κέντρο της κόρης
        # -------------------------------------------------

        M = cv2.moments(best)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])


        radius_draw = int(
            pupil_diameter / 2.0
        )


        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

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


    # Αν δεν βρεθεί έγκυρη κόρη
    return None, vis, eye_bin


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # -----------------------------------------------------
    # Mirror εικόνας
    # -----------------------------------------------------

    frame = cv2.flip(frame, 1)


    # -----------------------------------------------------
    # Grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # Face detection
    # -----------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )


    eye_debugs = []
    pupil_sizes = []


    # =====================================================
    # PROCESS FIRST DETECTED FACE
    # =====================================================

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # -------------------------------------------------
        # Χρησιμοποιείται μόνο το άνω 55% του προσώπου
        # -------------------------------------------------

        h_top = int(0.55 * h)

        roi_gray = gray[
            y:y + h_top,
            x:x + w
        ]

        roi_color = frame[
            y:y + h_top,
            x:x + w
        ]


        # -------------------------------------------------
        # Eye detection
        # -------------------------------------------------

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )


        # -------------------------------------------------
        # Ταξινόμηση από αριστερά προς δεξιά
        # και χρήση έως δύο ματιών
        # -------------------------------------------------

        eyes = sorted(
            eyes,
            key=lambda e: e[0]
        )[:2]


        # =================================================
        # PROCESS EACH EYE
        # =================================================

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
                detect_pupil_size(eye_roi)
            )


            if pupil_diameter is not None:

                # -----------------------------------------
                # Temporal smoothing ανά μάτι
                # -----------------------------------------

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


                # -----------------------------------------
                # Debug text
                # -----------------------------------------

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


        # Μόνο το πρώτο ανιχνευμένο πρόσωπο
        break


    # =====================================================
    # FUSION OF PUPIL DIAMETERS
    # =====================================================

    fused_diameter = None


    if len(pupil_sizes) == 2:

        fused_diameter = float(
            np.mean(pupil_sizes)
        )

        current_detection_type = "FUSED"


    elif len(pupil_sizes) == 1:

        fused_diameter = float(
            pupil_sizes[0]
        )

        current_detection_type = "SINGLE"


    else:

        current_detection_type = "NONE"


    # =====================================================
    # PREPARATION PHASE
    # =====================================================

    if not test_started:

        prep_elapsed = (
            time.time() - program_start_time
        )


        remaining = max(
            0,
            prep_duration - prep_elapsed
        )


        cv2.putText(
            frame,
            "PREPARE - STAY AT BASELINE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Test starts in: {remaining:.1f}s",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        if prep_elapsed >= prep_duration:

            # Καθαρισμός smoothing history ώστε
            # το πραγματικό test να ξεκινήσει καθαρά
            history_L.clear()
            history_R.clear()

            diameters.clear()

            total_frames = 0
            valid_frames = 0
            two_eye_frames = 0
            single_eye_frames = 0

            test_started = True
            test_start_time = time.time()

            beep(1000, 300)

            print("\n===================================")
            print("PUPIL STABILITY TEST STARTED")
            print("Stay still and keep looking at camera.")
            print("===================================\n")


    # =====================================================
    # STABILITY RECORDING PHASE
    # =====================================================

    elif not test_finished:

        elapsed = (
            time.time() - test_start_time
        )


        remaining = max(
            0,
            test_duration - elapsed
        )


        # -------------------------------------------------
        # Κάθε processed frame μετράει στο total
        # -------------------------------------------------

        total_frames += 1


        # -------------------------------------------------
        # Έγκυρη μέτρηση
        # -------------------------------------------------

        if fused_diameter is not None:

            diameters.append(
                fused_diameter
            )

            valid_frames += 1


            if current_detection_type == "FUSED":
                two_eye_frames += 1

            elif current_detection_type == "SINGLE":
                single_eye_frames += 1


        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------

        cv2.putText(
            frame,
            "STABILITY RECORDING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Time remaining: {remaining:.1f}s",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


        if fused_diameter is not None:

            cv2.putText(
                frame,
                f"Pupil diameter: {fused_diameter:.2f}px",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO VALID PUPIL",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


        # =================================================
        # TEST FINISHED
        # =================================================

        if elapsed >= test_duration:

            test_finished = True

            beep(1400, 500)


            print("\n===================================")
            print("PUPIL STABILITY TEST RESULTS")
            print("===================================")


            if len(diameters) > 0:

                mean_d = float(
                    np.mean(diameters)
                )

                std_d = float(
                    np.std(diameters)
                )


                # -----------------------------------------
                # Stability score
                # -----------------------------------------

                stability_score = max(
                    0.0,
                    1.0 - (std_d / sigma_max)
                )


                # -----------------------------------------
                # Detection rate
                # -----------------------------------------

                detection_rate = (
                    valid_frames / total_frames
                    if total_frames > 0
                    else 0.0
                )


                two_eye_rate = (
                    two_eye_frames / total_frames
                    if total_frames > 0
                    else 0.0
                )


                single_eye_rate = (
                    single_eye_frames / total_frames
                    if total_frames > 0
                    else 0.0
                )


                print(
                    f"Total frames: {total_frames}"
                )

                print(
                    f"Valid frames: {valid_frames}"
                )

                print(
                    f"Two-eye frames: {two_eye_frames}"
                )

                print(
                    f"Single-eye frames: {single_eye_frames}"
                )

                print(
                    f"Mean pupil diameter: {mean_d:.2f} px"
                )

                print(
                    f"Std deviation: {std_d:.3f} px"
                )

                print(
                    f"Detection rate: {detection_rate:.3f}"
                )

                print(
                    f"Two-eye detection rate: {two_eye_rate:.3f}"
                )

                print(
                    f"Single-eye detection rate: {single_eye_rate:.3f}"
                )

                print(
                    f"Stability score: {stability_score:.3f}"
                )


                print("\nFORMULA CHECK:")

                print(
                    "Stability = "
                    "max(0, 1 - std / sigma_max)"
                )

                print(
                    f"          = "
                    f"max(0, 1 - {std_d:.3f} / {sigma_max:.1f})"
                )

                print(
                    f"          = {stability_score:.3f}"
                )


            else:

                print(
                    "No valid pupil measurements "
                    "were collected."
                )


            print("===================================\n")

            break


    # =====================================================
    # DISPLAY WINDOWS
    # =====================================================

    cv2.imshow(
        "Pupil Stability Test",
        frame
    )


    for name, img in eye_debugs:

        cv2.imshow(
            name,
            img
        )


    # Προαιρετικός χειροκίνητος τερματισμός
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()