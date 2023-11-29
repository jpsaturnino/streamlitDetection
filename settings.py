from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
WEBCAM = 'Webcam'
RTSP = 'RTSP'

SOURCES_LIST = [WEBCAM, RTSP]

# Images config
IMAGES_DIR = ROOT / 'images'

# Videos config
VIDEO_DIR = ROOT / 'videos'

# ML Model config
MODEL_DIR = ROOT / 'weights'
MODEL_VERSION = 'v5'
DETECTION_MODEL = MODEL_DIR / MODEL_VERSION / 's.pt'
POSE_MODEL = MODEL_DIR / 'yolov8n-pose.pt'

# Webcam
WEBCAM_PATH = 0
