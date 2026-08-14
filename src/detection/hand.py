from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode.VIDEO

hand_option = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/mediapipe/hand_landmarker.task'),
    running_mode=VisionRunningMode,
    num_hands=2
)

hand_model = vision.HandLandmarker.create_from_options(
    hand_option
)