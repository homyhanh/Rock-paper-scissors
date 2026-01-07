import cv2
import mediapipe as mp
import csv
import os
import math
import time

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

def dist2d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0:
        return 0.0
    c = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(c))

def curl_angle(lm, mcp, pip, tip):
    # góc giữa (MCP->PIP) và (PIP->TIP) trên mặt phẳng ảnh
    v1 = (lm[pip].x - lm[mcp].x, lm[pip].y - lm[mcp].y)
    v2 = (lm[tip].x - lm[pip].x, lm[tip].y - lm[pip].y)
    return angle_between(v1, v2)

def extract_features(lm, handed_label):
    """
    Trả về 20 feature theo thứ tự:
    10 tọa độ tips (x4,y4, x8,y8, x12,y12, x16,y16, x20,y20),
    5 khoảng cách (4-8, 8-12, 12-16, 16-20, 4-20),
    5 curl (thumb/index/middle/ring/pinky).
    Normalize: dịch về WRIST(0), scale theo 0->9, mirror tay trái.
    """
    wrist = (lm[0].x, lm[0].y)
    middle_mcp = (lm[9].x, lm[9].y)
    scale = dist2d(wrist, middle_mcp) or 1.0

    pts = []
    for i in range(21):
        x = (lm[i].x - wrist[0]) / scale
        y = (lm[i].y - wrist[1]) / scale
        if handed_label == "Left":
            x = -x  # mirror tay trái
        pts.append((x, y))

    # 1) tips (x,y)
    feat = []
    tip_ids = [4, 8, 12, 16, 20]
    for tid in tip_ids:
        feat.extend([pts[tid][0], pts[tid][1]])

    # 2) distances giữa tips
    for a, b in [(4,8), (8,12), (12,16), (16,20), (4,20)]:
        feat.append(dist2d(pts[a], pts[b]))

    # 3) curl angles (dùng lm gốc để đo khớp)
    feat.append(curl_angle(lm, 2, 3, 4))      # thumb
    feat.append(curl_angle(lm, 5, 6, 8))      # index
    feat.append(curl_angle(lm, 9,10,12))      # middle
    feat.append(curl_angle(lm,13,14,16))      # ring
    feat.append(curl_angle(lm,17,18,20))      # pinky)

    return feat

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
