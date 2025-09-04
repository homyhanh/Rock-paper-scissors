import cv2
import mediapipe as mp
import joblib
import math
import numpy as np

# ====== CONFIG ======
MODEL_PATH = "model.pkl"
BACKGROUND_PATH = "image/background.png"
THRESHOLD  = 0.70
MAPPING    = {0:"Unknown", 1:"Scissors", 2:"Rock", 3:"Paper"}
LOSE_TO    = {1:3, 2:1, 3:2}

model = joblib.load(MODEL_PATH)
HAS_PROBA = hasattr(model, "predict_proba")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ---------- HÌNH HỌC ----------
def dist2d(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(v1[0], v1[1]); n2 = math.hypot(v2[0], v2[1])
    if n1 == 0 or n2 == 0: return 0.0
    c = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(c))

def curl_angle(lm, mcp, pip, tip):
    v1 = (lm[pip].x - lm[mcp].x, lm[pip].y - lm[mcp].y)
    v2 = (lm[tip].x - lm[pip].x, lm[tip].y - lm[pip].y)
    return angle_between(v1, v2)

# ---------- TRÍCH FEATURES ----------
def extract_features(lm, handed_label):
    wrist = (lm[0].x, lm[0].y)
    middle_mcp = (lm[9].x, lm[9].y)
    scale = dist2d(wrist, middle_mcp) or 1.0

    pts = []
    for i in range(21):
        x = (lm[i].x - wrist[0]) / scale
        y = (lm[i].y - wrist[1]) / scale
        if handed_label == "Left":
            x = -x
        pts.append((x, y))

    feat = []
    for tid in [4, 8, 12, 16, 20]:
        feat.extend([pts[tid][0], pts[tid][1]])

    for a, b in [(4,8), (8,12), (12,16), (16,20), (4,20)]:
        feat.append(dist2d(pts[a], pts[b]))

    feat.append(curl_angle(lm, 2, 3, 4))      # thumb
    feat.append(curl_angle(lm, 5, 6, 8))      # index
    feat.append(curl_angle(lm, 9,10,12))      # middle
    feat.append(curl_angle(lm,13,14,16))      # ring
    feat.append(curl_angle(lm,17,18,20))      # pinky
    return feat

# ---------- HÀM GHÉP ẢNH ----------
def overlay_on_bg(bg_bgr, fg_bgr, x, y, alpha=1.0, mask=None):
    """
    Ghép fg_bgr lên bg_bgr tại vị trí (x,y) với alpha (0..1).
    Nếu mask (uint8 0..255) được cung cấp (ví dụ bo góc), dùng mask đó để blend.
    """
    H, W = bg_bgr.shape[:2]
    h, w = fg_bgr.shape[:2]
    if x < 0 or y < 0 or x+w > W or y+h > H:
        # Cắt cho an toàn nếu vượt biên
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x+w), min(H, y+h)
        fg_crop = fg_bgr[y0-y:y1-y, x0-x:x1-x]
        if mask is not None:
            mask = mask[y0-y:y1-y, x0-x:x1-x]
        bg_roi = bg_bgr[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = x, y, x+w, y+h
        fg_crop = fg_bgr
        bg_roi = bg_bgr[y0:y1, x0:x1]

    if mask is None:
        cv2.addWeighted(fg_crop, alpha, bg_roi, 1-alpha, 0, bg_roi)
    else:
        # mask 0..255 -> tạo alpha map 0..1
        a = (mask.astype(np.float32)/255.0) * alpha
        a = a[..., None]  # (h,w,1)
        bg_roi[:] = (fg_crop.astype(np.float32)*a + bg_roi.astype(np.float32)*(1-a)).astype(np.uint8)

def rounded_rect_mask(w, h, r):
    """Tạo mask bo góc cho ROI (w×h), bán kính r."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (r,0), (w-r, h), 255, -1)
    cv2.rectangle(mask, (0,r), (w, h-r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w-r, r), r, 255, -1)
    cv2.circle(mask, (r, h-r), r, 255, -1)
    cv2.circle(mask, (w-r, h-r), r, 255, -1)
    return mask

# ---------- MAIN LOOP (DETECT -> COMPOSITE -> SHOW) ----------
cap = cv2.VideoCapture(0)

# Đọc nền một lần, xử lý alpha nếu có
bg = cv2.imread(BACKGROUND_PATH, cv2.IMREAD_UNCHANGED)
if bg is None:
    raise FileNotFoundError(f"Không đọc được ảnh nền: {BACKGROUND_PATH}")

# Nếu ảnh nền là 4 kênh (có alpha), tách alpha rồi đặt nền lên màu đen/trắng tùy thích
if bg.shape[2] == 4:
    bgr = bg[..., :3]
    alpha = bg[..., 3:4] / 255.0
    bg_canvas = (bgr.astype(np.float32)*alpha + 255*(1-alpha)).astype(np.uint8)  # nền trắng
else:
    bg_canvas = bg.copy()

# Lấy kích thước gốc của ảnh nền
TARGET_H, TARGET_W = bg_canvas.shape[:2]

# Kích thước khung camera ghép lên nền
CAM_W, CAM_H = 360, 240

with mp_hands.Hands(model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Vẽ landmarks trên chính frame
        image = frame.copy()
        shown_text = "Pred: -"
        color = (0,200,255)

        if results.multi_hand_landmarks:
            handed = [h.classification[0].label for h in (results.multi_handedness or [])]
            hand_lm = results.multi_hand_landmarks[0]
            label = handed[0] if handed else "Right"

            mp_drawing.draw_landmarks(
                image, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            feats = extract_features(hand_lm.landmark, label)
            if HAS_PROBA:
                proba = model.predict_proba([feats])[0]
                idx = int(proba.argmax())
                cls_label = int(model.classes_[idx])
                max_p = float(proba[idx])
                pred = cls_label if max_p >= THRESHOLD else 0
                shown_text = f"Pred: {MAPPING[pred]} ({max_p:.2f})"
                color = (0,255,0) if pred != 0 else (0,200,255)
            else:
                pred = int(model.predict([feats])[0])
                shown_text = f"Pred: {MAPPING.get(pred,'Unknown')}"
                color = (0,255,0)

        cv2.putText(image, "ESC to quit", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, shown_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # === GHÉP camera vào nền ===
        composed = bg_canvas.copy()

        # Resize frame camera
        cam = cv2.resize(image, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

        # Vị trí muốn đặt (trên nền)
        x, y = TARGET_W - CAM_W - 24, TARGET_H - CAM_H - 24  # góc phải dưới, cách viền 24px

        overlay_on_bg(composed, cam, x+6, y+6, alpha=0.6)

        cv2.imshow("RPS on Background", composed)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

cap.release()
cv2.destroyAllWindows()
