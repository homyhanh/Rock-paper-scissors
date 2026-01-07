import math
import cv2
import numpy as np

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

def paste_icon_noscale(bg, icon_rgba, x, y, max_size=270):
    """
    Dán icon RGBA lên bg tại (x,y).
    - Nếu icon lớn hơn max_size, tự động thu nhỏ theo tỉ lệ.
    - Giữ alpha trong suốt.
    """
    if icon_rgba is None or icon_rgba.shape[2] < 4:
        return

    H, W = bg.shape[:2]
    h, w = icon_rgba.shape[:2]

    # Thu nhỏ theo tỉ lệ nếu cần
    scale = min(max_size / w, max_size / h, 1.0)  # <=1: chỉ thu nhỏ, không phóng to
    new_w, new_h = int(w * scale), int(h * scale)
    if scale < 1.0:
        icon_rgba = cv2.resize(icon_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # Cắt nếu vượt biên
    if x+w > W: w = W-x
    if y+h > H: h = H-y
    if w <= 0 or h <= 0:
        return

    roi = bg[y:y+h, x:x+w]
    icon_crop = icon_rgba[0:h, 0:w]

    bgr   = icon_crop[..., :3].astype(np.float32)
    alpha = (icon_crop[..., 3:4].astype(np.float32)) / 255.0

    roi[:] = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)

def paste_icon_simple(
    bg,                      # BGR background
    icon,                    # BGRA (đọc bằng IMREAD_UNCHANGED)
    x, y,                    # toạ độ góc trên-trái để dán
    scale=None,              # tỉ lệ phóng/thu (vd: 0.7) – ưu tiên nếu truyền
    flip=False               # True -> lật ảnh theo trục dọc (gương)
):
    if icon is None:
        return
    if icon.shape[2] == 3:
        # nếu ảnh không có alpha, thêm alpha=255 để dán "đè"
        a = np.full((icon.shape[0], icon.shape[1], 1), 255, np.uint8)
        icon = np.concatenate([icon, a], axis=2)

    if flip:
        icon = cv2.flip(icon, 1)

    ih, iw = icon.shape[:2]

    # ---- resize toàn ảnh theo yêu cầu (giữ tỉ lệ) ----
    if scale is not None:
        nw, nh = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
        icon = cv2.resize(icon, (nw, nh), interpolation=cv2.INTER_AREA)

    ih, iw = icon.shape[:2]
    H, W   = bg.shape[:2]
    if x >= W or y >= H:
        return

    # cắt nếu tràn biên
    iw = min(iw, W - x)
    ih = min(ih, H - y)
    if iw <= 0 or ih <= 0:
        return
    icon = icon[:ih, :iw]
    roi  = bg[y:y+ih, x:x+iw]

    # alpha blend
    bgr   = icon[..., :3].astype(np.float32)
    alpha = (icon[..., 3:4].astype(np.float32)) / 255.0
    roi[:] = (bgr * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)



