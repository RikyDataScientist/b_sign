from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode.VIDEO

face_option = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/mediapipe/face_landmarker.task'),
    running_mode=VisionRunningMode
)

face_model = vision.FaceLandmarker.create_from_options(
    face_option
)