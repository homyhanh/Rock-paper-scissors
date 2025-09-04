import cv2
import mediapipe as mp
import csv
import os
import math
import time
from utils import *
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

CSV_PATH = "data.csv"
HEADERS = [
    # 1) fingertip normalized coords (x,y) for [4,8,12,16,20]
    "x_4","y_4","x_8","y_8","x_12","y_12","x_16","y_16","x_20","y_20",
    # 2) selected fingertip distances
    "d_4_8","d_8_12","d_12_16","d_16_20","d_4_20",
    # 3) curl angles (deg) for thumb/index/middle/ring/pinky
    "curl_thumb","curl_index","curl_middle","curl_ring","curl_pinky",
    # label
    "Label"  # 0=Unknown (tuỳ chọn), 1=Scissors, 2=Rock, 3=Paper
]

def ensure_csv(path, headers):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)


# ------------------------ MAIN ------------------------
ensure_csv(CSV_PATH, HEADERS)

cap = cv2.VideoCapture(0)

# chống ghi trùng liên tiếp & debounce
last_row = None
last_save_time = 0.0
MIN_INTERVAL = 0.25  # giãn cách tối thiểu 250ms

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:

    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        features = None

        # vẽ và trích feature
        image_bgr = image.copy()
        if results.multi_hand_landmarks:
            # ghép handedness ↔ landmarks cho chắc
            handed = [h.classification[0].label for h in (results.multi_handedness or [])]
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                label = handed[i] if i < len(handed) else "Right"
                features = extract_features(hand_landmarks.landmark, label)

        # UI
        cv2.putText(image_bgr, "0=Unknown | 1=Scissors | 2=Rock | 3=Paper | ESC=Quit",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("RPS Feature Collector", image_bgr)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # Ghi khi có hand + nhấn 0/1/2/3
        if features is not None and key in (ord('0'), ord('1'), ord('2'), ord('3')):
            if now - last_save_time >= MIN_INTERVAL:
                if key == ord('0'):
                    label = 0  # Unknown (tuỳ chọn)
                elif key == ord('1'):
                    label = 1  # Scissors
                elif key == ord('2'):
                    label = 2  # Rock
                else:
                    label = 3  # Paper

                row = features + [label]

                # chống trùng y hệt liên tiếp
                if row != last_row:
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(row)
                    print(f"Saved row (len={len(row)}): label={label}")
                    last_row = row
                    last_save_time = now
                else:
                    print("Skip duplicate immediate row.")

        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
