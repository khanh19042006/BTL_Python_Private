import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
from dotenv import load_dotenv

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Tắt failsafe pyautogui (cẩn thận, chỉ dùng trong test)
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01  # Giảm độ trễ nhỏ

# Biến EMA smoothing
prev_x = prev_y = None
smoothing_factor = 0.2

# --- KHỞI TẠO HAND LANDMARKER (Tasks API) ---
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

model_path = "hand_landmarker.task"  # Tự động tải về nếu chưa có (~30-50MB)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

# Danh sách kết nối landmarks tay (21 điểm) - chuẩn từ MediaPipe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),       # Ngón trỏ
    (5, 9), (9, 13), (13, 17), (0, 17),   # Lòng bàn tay
    (9, 10), (10, 11), (11, 12),          # Ngón giữa
    (13, 14), (14, 15), (15, 16),         # Ngón áp út
    (17, 18), (18, 19), (19, 20)          # Ngón út
]

def draw_hand_landmarks(frame, hand_landmarks):
    """Vẽ 21 landmarks + đường nối thủ công bằng OpenCV (không dùng drawing_utils)"""
    if not hand_landmarks:
        return

    h, w, _ = frame.shape

    # Vẽ các điểm (xanh lá)
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    # Vẽ đường nối (đỏ)
    for conn in HAND_CONNECTIONS:
        start_idx, end_idx = conn
        if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]
            x1, y1 = int(start_lm.x * w), int(start_lm.y * h)
            x2, y2 = int(end_lm.x * w), int(end_lm.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def count_fingers(landmarks_list):
    """Đếm số ngón tay giơ lên (logic tương tự code cũ)"""
    if not landmarks_list:
        return 0
    landmarks = landmarks_list[0]  # Chỉ xử lý 1 tay

    tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    fingers = []

    # Ngón cái: dùng x (phù hợp khi tay hướng phải/trái)
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 ngón còn lại: dùng y
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def main():
    global prev_x, prev_y

    print("=" * 60)
    print("HAND TRACKING MOUSE CONTROL - MediaPipe Tasks API (2026 compatible)")
    print("Di chuyển chuột bằng ngón trỏ")
    print("Giơ đúng 1 ngón tay để click trái")
    print("Nhấn 'q' để thoát")
    print("=" * 60)

    load_dotenv()
    droicam = os.getenv("URL_SCREEN")
    if not droicam:
        print("Lỗi: Chưa có URL_SCREEN")
        exit()
    cap = cv2.VideoCapture(droicam)
    if not cap.isOpened():
        print("Lỗi: Không mở được camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_frame_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame.")
            break

        frame = cv2.flip(frame, 1)  # Lật ngang (gương)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        finger_count = 0

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks)

                # Lấy tọa độ đầu ngón trỏ (landmark 8)
                index_tip = hand_landmarks[8]

                screen_w, screen_h = pyautogui.size()
                margin = 0.1
                nav_start_x = screen_w * margin
                nav_end_x   = screen_w * (1 - margin)
                nav_start_y = screen_h * margin
                nav_end_y   = screen_h * (1 - margin)

                target_x = np.interp(index_tip.x, [0, 1], [nav_start_x, nav_end_x])
                target_y = np.interp(index_tip.y, [0, 1], [nav_start_y, nav_end_y])

                # EMA làm mượt chuyển động
                if prev_x is None:
                    prev_x, prev_y = target_x, target_y
                else:
                    prev_x = smoothing_factor * target_x + (1 - smoothing_factor) * prev_x
                    prev_y = smoothing_factor * target_y + (1 - smoothing_factor) * prev_y

                pyautogui.moveTo(int(prev_x), int(prev_y), duration=0)

                # Đếm ngón & click nếu = 1
                finger_count = count_fingers(result.hand_landmarks)

                if finger_count == 1:
                    pyautogui.click()
                    time.sleep(0.15)  # Tránh click liên tục

        # Hiển thị số ngón + FPS
        cv2.putText(frame, f"Fingers: {finger_count}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_frame_time) if curr_time > prev_frame_time else 0
        prev_frame_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Hand Mouse Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Chương trình đã dừng. Tài nguyên đã giải phóng.")


if __name__ == "__main__":
    main()