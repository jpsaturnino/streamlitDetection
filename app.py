# Python In-built packages
import os
from pathlib import Path
from functools import partial
import zipfile

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

# Sidebar operation selection
operation = st.sidebar.selectbox("Operação", ["Vigilância", "Avaliação"])

if operation.lower() == "vigilância":
    # Main page heading
    st.title("Vigilância")
    
    # Sidebar
    st.sidebar.header("Configurar Modelo")

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Detection', 'Pose Estimation + Detection'])

    conf_detection = 0
    conf_pose = 0
    model_path = ''
    models = {
        'DETECTION': None,
        'DETECTION_POSE': None
    }

    # Selecting model
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
        confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
    elif model_type == 'Pose Estimation + Detection':
        model_path = Path(settings.POSE_MODEL)
        confidence = float(st.sidebar.slider("Select Detection Confidence", 25, 100, 40)) / 100
        # conf_pose = float(st.sidebar.slider("Select Pose Confidence", 25, 100, 40)) / 100

    # Load Pre-trained ML Model
    try:
        models["DETECTION"] = helpers.load_model(settings.DETECTION_MODEL)
        logger.info("Detection Model Loaded Successfully")

        if(model_type == 'Pose Estimation + Detection'):
            models["DETECTION_POSE"] = helpers.load_model(settings.POSE_MODEL)
            logger.info("Pose Estimation Model Loaded Successfully")

    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    # Displaying the video feed
    video_frame_callback = partial(helpers.play_webcam, conf = confidence, models = models)

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif operation.lower() == 'avaliação':
    # Main page heading
    st.title("Avaliação")
    
    # Sidebar
    st.sidebar.header("Configurar Avaliação")

    custom_db = st.sidebar.checkbox("Base de dados personalizada?")
    with_pose = st.sidebar.checkbox("Aplicar estimador de pose?")

    if custom_db:
        uploaded_file = st.sidebar.file_uploader("Selecionar Base de Dados:")

        if uploaded_file:
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall("temp")
                for file in os.walk("temp"):
                    st.write(file)

    model = st.sidebar.selectbox("Modelo", ["YOLOv5", "YOLOv8"])
    model = st.sidebar.selectbox("Versão", ["nano", "small", "large"])
    
    st.sidebar.button("Iniciar")