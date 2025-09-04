import cv2
import time
import math
import numpy as np
import joblib
import mediapipe as mp
import os
from utils import *  # dùng extract_features, paste_icon_noscale

# ====== CONFIG ======
BACKGROUND_PATH = "image/background.png"
MODEL_PATH      = "model.pkl"
THRESHOLD       = 0.8
MAPPING         = {0: "unknown", 1: "scissors", 2: "rock", 3: "paper"}
LOSE_TO         = {"scissors":"paper", "paper":"rock", "rock":"scissors"}

# Vị trí icon
PLAYER_X, PLAYER_Y = 600, 200
COMPUTER_X, COMPUTER_Y = 120, 200

# Camera box (góc dưới phải)
CAM_W, CAM_H = 350, 200
MARGIN = 24

# Icon
ICON_PLAYER = {
    "rock":      "image/player/rock.png",
    "paper":     "image/player/paper.png",
    "scissors":  "image/player/scissors.png",
}
ICON_AI = {
    "rock": "image/computer/rock",
    "paper": "image/computer/paper",
    "scissors": "image/computer/scissors"
}

# ====== LOADS ======
bg = cv2.imread(BACKGROUND_PATH)
if bg is None:
    raise FileNotFoundError(f"Không đọc được ảnh nền: {BACKGROUND_PATH}")
hBG, wBG = bg.shape[:2]

# toạ độ camera
x1_cam = max(0, wBG - CAM_W - MARGIN)
y1_cam = max(0, hBG - CAM_H - MARGIN)
x2_cam = min(wBG, x1_cam + CAM_W)
y2_cam = min(hBG, y1_cam + CAM_H)

# model
model = joblib.load(MODEL_PATH)
HAS_PROBA = hasattr(model, "predict_proba")

# ====== STATES ======
STATE_DETECT, STATE_SHOW = 0, 1
state = STATE_DETECT
last_label = None
last_prob  = 0.0

# Countdown 3s trước khi detect
countdown_on   = True
initialTime    = time.time()  # mốc bắt đầu đếm 3s

# Lịch sử tay đã LOCK (giữ thứ tự) để kiểm tra trùng với tay cuối
user_labels = []  # ví dụ: ["rock", "paper", ...]

# ====== MEDIAPIPE ======
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles= mp.solutions.drawing_styles
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ====== CAMERA ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        imgBG = bg.copy()
        cam_panel = frame.copy()

        if state == STATE_DETECT:
            # --- Đếm ngược 3s (không detect trong thời gian này) ---
            if countdown_on:
                timer = time.time() - initialTime
                cv2.putText(imgBG, str(max(1, 3-int(timer))), (440, 350),
                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 4)
                if timer >= 3.0:
                    countdown_on = False  # hết đếm, bắt đầu detect
            else:
                # --- Detect tay ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                if res.multi_hand_landmarks:
                    hand_lm = res.multi_hand_landmarks[0]
                    handed_label = "Right"
                    if res.multi_handedness:
                        handed_label = res.multi_handedness[0].classification[0].label

                    mp_draw.draw_landmarks(
                        cam_panel, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    feats = extract_features(hand_lm.landmark, handed_label)

                    # Dự đoán
                    if HAS_PROBA:
                        proba = model.predict_proba([feats])[0]
                        idx = int(np.argmax(proba))
                        cls = int(model.classes_[idx]) if hasattr(model, "classes_") else idx
                        p = float(proba[idx])
                    else:
                        cls = int(model.predict([feats])[0])
                        p = 1.0

                    label = MAPPING.get(cls, "unknown")

                    # Chỉ xét nếu là tay hợp lệ và vượt ngưỡng
                    if label in ("rock", "paper", "scissors") and p >= THRESHOLD:
                        # Kiểm tra trùng với tay của LƯỢT TRƯỚC (cái cuối)
                        if user_labels and label == user_labels[-1]:
                            cv2.putText(cam_panel, f'"{label}" bi trung voi luot truoc - hay doi tay',
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            last_label = label
                            last_prob  = p
                            user_labels.append(label)
                            state = STATE_SHOW

            cv2.putText(cam_panel, "Detecting... (press R to reset)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        else:  # STATE_SHOW
            info = f"LOCKED: {last_label} (p={last_prob:.2f}) - Press R to reset"
            cv2.putText(cam_panel, info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

            if last_label in ("rock", "paper", "scissors"):
                icon_path = ICON_PLAYER[last_label]
                icon_rgba = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                paste_icon_noscale(imgBG, icon_rgba, PLAYER_X, PLAYER_Y)

                # PC ra tay thua
                icon_lose_path = ICON_PLAYER[LOSE_TO[last_label]]  
                icon_lose_rgba = cv2.imread(icon_lose_path, cv2.IMREAD_UNCHANGED)
                paste_icon_noscale(imgBG, icon_lose_rgba, COMPUTER_X, COMPUTER_Y)

        # Ghép camera xuống góc dưới-phải
        cam_resized = cv2.resize(cam_panel, (x2_cam - x1_cam, y2_cam - y1_cam),
                                 interpolation=cv2.INTER_AREA)
        imgBG[y1_cam:y2_cam, x1_cam:x2_cam] = cam_resized

        cv2.imshow("ROCK, PAPER, SCISSORS", imgBG)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        if key in (ord('r'), ord('R')):
            # Reset: quay lại đếm 3s, cho phép dùng lại lịch sử từ đầu
            state = STATE_DETECT
            last_label = None
            last_prob  = 0.0
            countdown_on = True
            initialTime  = time.time()

finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
