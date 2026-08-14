from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode.VIDEO

pose_option = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/mediapipe/pose_landmarker.task'),
    running_mode=VisionRunningMode
)

pose_model = vision.PoseLandmarker.create_from_options(
    pose_option
)