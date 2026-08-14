import cv2

def draw_face(frame, landmarks):
    h, w, _ = frame.shape

    for landmark in landmarks:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            2,
            (255, 0, 255),
            -1,
            cv2.LINE_AA
        )

def draw_landmarks(frame, landmarks, connections, landmark_color, line_color):
    h, w, _ = frame.shape

    # Draw connections
    for start, end in connections:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)

        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            line_color,
            2,
            cv2.LINE_AA
        )

    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)

        cv2.circle(
            frame,
            (x, y),
            5,
            landmark_color,
            -1,
            cv2.LINE_AA
        )