import numpy as np

def extract_keypoints(face_res, hand_res, pose_res):
    face = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in face_res.face_landmarks[0]
    ]).flatten() if face_res.face_landmarks else np.zeros(478 * 3)

    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    for landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
        keypoints = np.array([
            [landmark.x, landmark.y, landmark.z] for landmark in landmarks
        ]).flatten()
        label = handedness[0].category_name
        if label == 'Left':
            lh = keypoints
        elif label == 'Right':
            rh = keypoints

    pose = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in pose_res.pose_landmarks[0]
    ]).flatten() if pose_res.pose_landmarks else np.zeros(33 * 3)

    return np.concatenate([face, lh, rh, pose])
