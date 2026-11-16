import cv2
import mediapipe as mp
import math
import time
import random

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "models/face_landmarker.task"

EAR_THRESHOLD = 0.20
BLINK_FRAMES = 3

CHALLENGE_TIME = 5


# =========================================================
# EAR CALCULATION
# =========================================================

def calculate_ear(eye_points):

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = math.dist(p2, p6)
    vertical_2 = math.dist(p3, p5)

    horizontal = math.dist(p1, p4)

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2 * horizontal)


# =========================================================
# MEDIAPIPE
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
# CAMERA
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# =========================================================
# EYE LANDMARKS
# =========================================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# =========================================================
# CHALLENGE
# =========================================================

required_blinks = random.choice([2, 3])

blink_count = 0

blink_counter = 0

challenge_start = time.time()

liveness_confirmed = False

challenge_failed = False


# =========================================================
# TIMESTAMP
# =========================================================

frame_timestamp_ms = 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read webcam.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


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
    # FACE FOUND
    # =====================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]


        # -------------------------------------------------
        # Convert eye landmarks
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
        # EAR
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


        # =================================================
        # BLINK DETECTION
        # =================================================

        if not liveness_confirmed and not challenge_failed:

            if average_ear < EAR_THRESHOLD:

                blink_counter += 1

            else:

                # Eyes have reopened
                if blink_counter >= BLINK_FRAMES:

                    blink_count += 1

                blink_counter = 0


        # =================================================
        # CHECK CHALLENGE
        # =================================================

        if blink_count >= required_blinks:

            liveness_confirmed = True


        # =================================================
        # TIMEOUT
        # =================================================

        elapsed_time = time.time() - challenge_start

        remaining_time = max(
            0,
            CHALLENGE_TIME - int(elapsed_time)
        )


        if (
            elapsed_time > CHALLENGE_TIME
            and not liveness_confirmed
        ):

            challenge_failed = True


        # =================================================
        # DISPLAY STATUS
        # =================================================

        cv2.putText(
            frame,
            "LIVENESS CHECK",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


        if liveness_confirmed:

            cv2.putText(
                frame,
                "LIVENESS CONFIRMED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Ready for face recognition",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


        elif challenge_failed:

            cv2.putText(
                frame,
                "CHALLENGE FAILED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Press R to retry",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                f"BLINK {required_blinks} TIMES",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )

            cv2.putText(
                frame,
                f"Completed: {blink_count}/{required_blinks}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Time: {remaining_time}s",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # =================================================
        # EAR DISPLAY
        # =================================================

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (width - 180, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
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
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Face Attendance - Active Liveness",
        frame
    )


    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # Q = quit
    if key == ord("q"):

        break


    # R = restart challenge
    if key == ord("r"):

        required_blinks = random.choice([2, 3])

        blink_count = 0

        blink_counter = 0

        challenge_start = time.time()

        liveness_confirmed = False

        challenge_failed = False


# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Program closed.")
