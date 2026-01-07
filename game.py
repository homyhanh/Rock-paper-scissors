import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from utils import *  # extract_features, paste_icon_noscale

# ============ CONFIG ============
MENU_PATH       = "image/background.png"
BACKGROUND_PATH = "image/background_play.png"
ICON_UNKNOWN_PATH   = "image/unknown.png"
DUPLICATE_CHAT_PATH = "image/duplicate_hand.png"
UNKNOWN_CHAT_PATH   = "image/unknown_chat.png"
MODEL_PATH      = "model.pkl"
MAPPING         = {0: "unknown", 1: "scissors", 2: "rock", 3: "paper"}
LOSE_TO         = {"scissors": "paper", "paper": "rock", "rock": "scissors"}
# ================================

# Vị trí icon
PLAYER_X, PLAYER_Y     = 600, 240
COMPUTER_X, COMPUTER_Y = 130, 240
MESSAGE_X, MESSAGE_Y   = 300, 420
UNKNOWN_X, UNKNOWN_Y   = 560, 240

# Camera box
CAM_W, CAM_H = 300, 200
MARGIN_X, MARGIN_Y = 80, 24

# Icon
ICON = {
    "rock":      "image/rock.png",
    "paper":     "image/paper.png",
    "scissors":  "image/scissors.png",
}

# ====== LOAD BG/MENU ======
bg = cv2.imread(BACKGROUND_PATH)
hBG, wBG = bg.shape[:2]

x1_cam = max(0, wBG - CAM_W - MARGIN_X)
y1_cam = max(0, hBG - CAM_H - MARGIN_Y)
x2_cam = min(wBG, x1_cam + CAM_W)
y2_cam = min(hBG, y1_cam + CAM_H)

# Load model
model, feature_names = joblib.load(MODEL_PATH)
current_label = None

# ====== MENU ======
HEADER = "ROCK, PAPER, SCISSORS"
menu_img = cv2.imread(MENU_PATH)
# Búa nhỏ hơn kéo/bao
SCALE_MAP = {"rock": 0.85, "paper": 1.10, "scissors": 1.00}

while True:
    cv2.imshow(HEADER, menu_img)
    k = cv2.waitKey(16) & 0xFF
    if k in (ord('s'), ord('S')): break
    if k == 27: cv2.destroyAllWindows(); raise SystemExit

# ====== INIT MEDIAPIPE + CAMERA ======
mp_hands = mp.solutions.hands
mp_draw, mp_styles = mp.solutions.drawing_utils, mp.solutions.drawing_styles
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load icons/chat
icon_unknown = cv2.imread(ICON_UNKNOWN_PATH, cv2.IMREAD_UNCHANGED)
unknown_chat   = cv2.imread(UNKNOWN_CHAT_PATH, cv2.IMREAD_UNCHANGED)

try:
    while True:
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)

        imgBG = bg.copy()
        cam_panel = frame.copy()

        # Nhận diện tay
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand_lm = res.multi_hand_landmarks[0]
            handed_label = "Right"
            if res.multi_handedness:
                handed_label = res.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(cam_panel, hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style())

            feats = extract_features(hand_lm.landmark, handed_label)
            feats = pd.DataFrame([feats], columns=feature_names)
            
            cls = int(model.predict(feats)[0])
            current_label = MAPPING.get(cls)

            # ================= DETECT =================
            if current_label in ("rock", "paper", "scissors"):
                # Người chơi
                icon_rgba = cv2.imread(ICON[current_label], cv2.IMREAD_UNCHANGED)
                paste_icon_simple(imgBG, icon_rgba, PLAYER_X, PLAYER_Y, scale=0.4) 

                # Máy ra tay thua
                lose_label = LOSE_TO[current_label]
                lose_rgba = cv2.imread(ICON[lose_label], cv2.IMREAD_UNCHANGED)
                paste_icon_simple(imgBG, lose_rgba, COMPUTER_X, COMPUTER_Y,  scale=0.4, flip=1) 
            else:
                # Không nhận diện được
                icon_unknown_rgba = cv2.imread(ICON_UNKNOWN_PATH, cv2.IMREAD_UNCHANGED)
                paste_icon_noscale(imgBG, icon_unknown_rgba, UNKNOWN_X, UNKNOWN_Y) 
                paste_icon_noscale(imgBG, unknown_chat, MESSAGE_X, MESSAGE_Y) 

        # Camera overlay
        cam_resized = cv2.resize(cam_panel, (x2_cam-x1_cam, y2_cam-y1_cam))
        imgBG[y1_cam:y2_cam, x1_cam:x2_cam] = cam_resized

        cv2.imshow(HEADER, imgBG)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break


finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

