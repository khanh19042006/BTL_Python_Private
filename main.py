import cv2
from dotenv import load_dotenv
import os
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

load_dotenv()

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = "hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    USE_NEW_API = True
except ImportError as e:
    exit(1)

# nếu bạn dùng cam máy tính thì droidcamp = 0
droidcamp = os.getenv("URL_SCREEN")

#Kết nối camera
def connect_to_droidcam():
    return cv2.VideoCapture(droidcamp)

def detect_hands_new_api(frame):
    # Chuyển BGR sang RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Phát hiện bàn tay
    detection_result = hand_landmarker.detect(mp_image)

    # Vẽ landmarks
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Vẽ các điểm landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Vẽ connections (21 điểm)
            connections = [
                [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
                [0, 5], [5, 6], [6, 7], [7, 8],  # Index
                [5, 9], [9, 10], [10, 11], [11, 12],  # Middle
                [9, 13], [13, 14], [14, 15], [15, 16],  # Ring
                [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]  # Pinky
            ]

            for connection in connections:
                start_idx, end_idx = connection
                start_point = (int(hand_landmarks[start_idx].x * frame.shape[1]),
                               int(hand_landmarks[start_idx].y * frame.shape[0]))
                end_point = (int(hand_landmarks[end_idx].x * frame.shape[1]),
                             int(hand_landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Đếm số ngón tay
            finger_count = count_fingers_new_api(hand_landmarks, detection_result.handedness[
                detection_result.hand_landmarks.index(hand_landmarks)])

            # Hiển thị số ngón tay
            h, w = frame.shape[:2]
            x = int(hand_landmarks[0].x * w)
            y = int(hand_landmarks[0].y * h)
            cv2.putText(frame, f"Fingers: {finger_count}", (x - 50, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detection_result

def count_fingers_new_api(hand_landmarks, handedness):
    """Đếm số ngón tay"""
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    is_right = handedness[0].category_name == "Right"

    # Thumb
    if is_right:
        if hand_landmarks[tip_ids[0]].x > hand_landmarks[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if hand_landmarks[tip_ids[0]].x < hand_landmarks[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Các ngón khác
    for id in range(1, 5):
        if hand_landmarks[tip_ids[id]].y < hand_landmarks[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

def main():
    print("=" * 50)
    print("HAND DETECTION VỚI DROIDCAM")
    print("=" * 50)

    # Kết nối đến DroidCam
    cap = connect_to_droidcam()
    if cap is None:
        return

    # Thiết lập độ phân giải và FPS (giam de giam do tre)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 24)
    # Thu nho buffer de giam do tre tu stream
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\nNhấn 'q' để thoát")
    print("Đang xử lý video...\n")

    frame_count = 0
    fps_start_time = cv2.getTickCount()

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Không thể đọc frame từ DroidCam")
                break

            # Nhận diện bàn tay
            processed_frame, results = detect_hands_new_api(frame)

            # Tính FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                fps = 30.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time

                if USE_NEW_API:
                    hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
                else:
                    hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

                print(f"FPS: {fps:.2f} | Hands detected: {hand_count}")

            # Hiển thị số bàn tay
            if USE_NEW_API:
                hand_count = len(results.hand_landmarks) if results.hand_landmarks else 0
            else:
                hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

            cv2.putText(processed_frame, f"Hands: {hand_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Hiển thị frame
            cv2.imshow("Hand Detection - DroidCam", processed_frame)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nĐang dừng...")
    finally:
        # Giải phóng tài nguyên
        cap.release()
        cv2.destroyAllWindows()
        if USE_NEW_API:
            hand_landmarker.close()
        print("Đã đóng kết nối.")

if __name__ == "__main__":
    main()