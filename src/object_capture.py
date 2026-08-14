import cv2, os
import mediapipe as mp
import numpy as np
from src.detection.face import face_model
from src.detection.hand import hand_model
from src.detection.pose import pose_model
from src.visualization.landmark import draw_face, draw_landmarks
from src.preprocessing.extraction import extract_keypoints

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]

POSE_CONNECTIONS = [
    # Face/head
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28),

    # Feet
    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32)
]

# Video Capture
cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    hand_result = hand_model.detect_for_video(
        mp_image,
        timestamp_ms
    )

    face_result = face_model.detect_for_video(
        mp_image,
        timestamp_ms
    )

    pose_result = pose_model.detect_for_video(
        mp_image,
        timestamp_ms
    )

    for hand_landmarks in hand_result.hand_landmarks:
        draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
            landmark_color=(255, 255, 255),
            line_color=(255, 200, 0)
        )

    for pose_landmarks in pose_result.pose_landmarks:
        draw_landmarks(
            frame,
            pose_landmarks,
            POSE_CONNECTIONS,
            landmark_color=(255, 255, 255),
            line_color=(255, 100, 0)
        )

    for face_landmarks in face_result.face_landmarks:

        draw_face(
            frame,
            face_landmarks
        )

    timestamp_ms += 33

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()