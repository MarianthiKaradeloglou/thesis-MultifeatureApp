import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# smoothing μόνο για size
history_L = deque(maxlen=5)
history_R = deque(maxlen=5)

FIX_W, FIX_H = 120, 80

def preprocess_eye(eye_roi):
    # Κρατάμε κυρίως τη μεσαία-κάτω περιοχή του ματιού
    h, w = eye_roi.shape
    y1 = int(0.20 * h)
    y2 = int(0.85 * h)
    x1 = int(0.10 * w)
    x2 = int(0.90 * w)
    eye_roi = eye_roi[y1:y2, x1:x2]

    eye_roi = cv2.resize(eye_roi, (FIX_W, FIX_H), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eye_eq = clahe.apply(eye_roi)

    eye_blur = cv2.GaussianBlur(eye_eq, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        eye_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return eye_roi, thresh


def detect_pupil_size(eye_roi):
    eye_gray, eye_bin = preprocess_eye(eye_roi)

    contours, _ = cv2.findContours(
        eye_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = -1

    roi_h, roi_w = eye_gray.shape
    cx_roi = roi_w / 2.0
    cy_roi = roi_h / 2.0

    for c in contours:
        area = cv2.contourArea(c)

        # λογικά όρια μεγέθους
        if area < 40 or area > 1200:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)

        # κέντρο contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # απόσταση από κέντρο ROI
        dist = np.sqrt((cx - cx_roi) ** 2 + (cy - cy_roi) ** 2)

        # φίλτρα σχήματος
        if aspect < 0.5 or aspect > 2.0:
            continue

        if circularity < 0.35:
            continue

        # score: προτιμάμε σχετικά κυκλικό contour κοντά στο κέντρο
        score = (2.0 * circularity) + (area / 500.0) - (dist / 100.0)

        if score > best_score:
            best_score = score
            best = c

    vis = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)

    if best is not None:
        area = cv2.contourArea(best)

        # equivalent diameter από εμβαδό contour
        pupil_diameter = 2.0 * np.sqrt(area / np.pi)

        M = cv2.moments(best)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        radius_draw = int(pupil_diameter / 2.0)

        cv2.circle(vis, (cx, cy), radius_draw, (0, 0, 255), 2)
        cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)

        return pupil_diameter, vis, eye_bin

    return None, vis, eye_bin


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eye_debugs = []
    pupil_sizes = []

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        h_top = int(0.55 * h)
        roi_gray = gray[y:y + h_top, x:x + w]
        roi_color = frame[y:y + h_top, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )

        # ως 2 μάτια, από αριστερά προς δεξιά
        eyes = sorted(eyes, key=lambda e: e[0])[:2]

        for i, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            if eye_roi.size == 0:
                continue

            pupil_diameter, eye_vis, eye_bin = detect_pupil_size(eye_roi)

            if pupil_diameter is not None:
                if i == 0:
                    history_L.append(pupil_diameter)
                    smooth_d = float(np.mean(history_L))
                else:
                    history_R.append(pupil_diameter)
                    smooth_d = float(np.mean(history_R))

                pupil_sizes.append(smooth_d)

                # debug text
                cv2.putText(
                    roi_color,
                    f"d={smooth_d:.1f}",
                    (ex, ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )

            eye_debugs.append((f"Eye {i} Gray", eye_vis))
            eye_debugs.append((f"Eye {i} Bin", eye_bin))

        break  # μόνο πρώτο πρόσωπο

    # fusion μόνο στο μέγεθος
    if len(pupil_sizes) == 2:
        fused_diameter = float(np.mean(pupil_sizes))
        print(f"FUSED pupil_diameter = {fused_diameter:.2f}")
    elif len(pupil_sizes) == 1:
        fused_diameter = pupil_sizes[0]
        print(f"SINGLE pupil_diameter = {fused_diameter:.2f}")
    else:
        print("NO PUPIL DETECTION")

    cv2.imshow("Frame", frame)

    for name, img in eye_debugs:
        cv2.imshow(name, img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()