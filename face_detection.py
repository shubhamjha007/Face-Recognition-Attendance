import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. MediaPipe Face Landmarker
# ==========================================

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# ==========================================
# 2. Open Webcam
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# ==========================================
# 3. Variables
# ==========================================

frame_timestamp_ms = 0
previous_time = time.time()


# ==========================================
# 4. Main Loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror camera like a normal webcam
    frame = cv2.flip(frame, 1)

    # OpenCV BGR → MediaPipe RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must continuously increase
    frame_timestamp_ms += 33

    # Run Face Landmarker
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # ======================================
    # 5. Process Face
    # ======================================

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            height, width, _ = frame.shape

            # --------------------------------
            # Draw facial landmarks
            # --------------------------------

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )


            # --------------------------------
            # Calculate bounding box
            # --------------------------------

            x_coordinates = [
                int(landmark.x * width)
                for landmark in face_landmarks
            ]

            y_coordinates = [
                int(landmark.y * height)
                for landmark in face_landmarks
            ]

            x_min = max(0, min(x_coordinates))
            y_min = max(0, min(y_coordinates))

            x_max = min(width, max(x_coordinates))
            y_max = min(height, max(y_coordinates))


            # --------------------------------
            # Draw bounding box
            # --------------------------------

            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )


            # --------------------------------
            # Face detected text
            # --------------------------------

            cv2.putText(
                frame,
                "FACE DETECTED",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ======================================
    # 6. FPS
    # ======================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # 7. Display
    # ======================================

    cv2.imshow(
        "Face Attendance System - Phase 1",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 8. Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Program closed.")
