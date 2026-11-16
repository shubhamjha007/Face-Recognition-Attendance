import cv2
import mediapipe as mp
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/face_landmarker.task"

# Blink threshold
EAR_THRESHOLD = 0.20

# Number of consecutive frames required
BLINK_FRAMES = 3


# =========================================================
# EAR CALCULATION
# =========================================================

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio.

    eye_points:
        [p1, p2, p3, p4, p5, p6]

    Formula:

        EAR = (|p2-p6| + |p3-p5|)
              ---------------------
                    2|p1-p4|
    """

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0

    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

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

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


# =========================================================
# VARIABLES
# =========================================================

frame_timestamp_ms = 0

blink_counter = 0
total_blinks = 0

previous_time = time.time()


# =========================================================
# MEDIAPIPE EYE LANDMARK INDICES
# =========================================================

# Left eye
LEFT_EYE = [
    33,    # p1
    160,   # p2
    158,   # p3
    133,   # p4
    153,   # p5
    144    # p6
]

# Right eye
RIGHT_EYE = [
    362,   # p1
    385,   # p2
    387,   # p3
    263,   # p4
    373,   # p5
    380    # p6
]


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp
    frame_timestamp_ms += 33

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]


        # -------------------------------------------------
        # Convert eye landmarks to pixel coordinates
        # -------------------------------------------------

        left_eye_points = []

        for index in LEFT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            left_eye_points.append((x, y))


        right_eye_points = []

        for index in RIGHT_EYE:

            landmark = face_landmarks[index]

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            right_eye_points.append((x, y))


        # -------------------------------------------------
        # Calculate EAR
        # -------------------------------------------------

        left_ear = calculate_ear(
            left_eye_points
        )

        right_ear = calculate_ear(
            right_eye_points
        )

        average_ear = (
            left_ear + right_ear
        ) / 2


        # -------------------------------------------------
        # Draw eye points
        # -------------------------------------------------

        for point in left_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 255),
                -1
            )


        for point in right_eye_points:

            cv2.circle(
                frame,
                point,
                3,
                (0, 255, 255),
                -1
            )


        # -------------------------------------------------
        # BLINK DETECTION
        # -------------------------------------------------

        if average_ear < EAR_THRESHOLD:

            blink_counter += 1

        else:

            # Eyes opened after being closed
            if blink_counter >= BLINK_FRAMES:

                total_blinks += 1

            blink_counter = 0


        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        if average_ear < EAR_THRESHOLD:

            status = "EYES CLOSED"

        else:

            status = "EYES OPEN"


        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Blinks: {total_blinks}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # -------------------------------------------------
        # LIVENESS STATUS
        # -------------------------------------------------

        if total_blinks > 0:

            liveness_text = "LIVENESS CONFIRMED"

            cv2.putText(
                frame,
                liveness_text,
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "BLINK TO VERIFY LIVENESS",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
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


    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    fps = 1 / (current_time - previous_time)

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (width - 120, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Face Attendance - Phase 2 Liveness",
        frame
    )


    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Program closed.")
