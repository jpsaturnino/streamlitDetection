# Python In-built packages
from pathlib import Path
from functools import partial

# External packages
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Local Modules
import settings
import helpers
import logging

logger = logging.getLogger(__name__)

# Setting page layout
st.set_page_config(
    page_title="PEwOBJ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("PEwOBJ")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Pose Estimation + Detection'])

conf_detection = 0
conf_pose = 0
model_path = ''

# Selecting model
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
elif model_type == 'Pose Estimation + Detection':
    model_path = Path(settings.POSE_MODEL)
    conf_detection = float(st.sidebar.slider("Select Detection Confidence", 25, 100, 40)) / 100
    conf_pose = float(st.sidebar.slider("Select Pose Confidence", 25, 100, 40)) / 100

# Load Pre-trained ML Model
try:
    model = helpers.load_model(model_path)
    logger.info("Model Loaded Successfully")
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Displaying the video feed
video_frame_callback = partial(helpers.play_webcam, conf = conf_detection, model = model)

webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=helpers.play_webcam,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)