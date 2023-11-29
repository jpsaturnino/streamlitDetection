import logging

from ultralytics import YOLO
import av
import cv2
import streamlit as st

logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def play_webcam(frame: av.VideoFrame):
    """
    Plays a webcam stream. Detects Objects in real-time using the object detection model.

    Parameters:
        conf: Confidence of the model.
        model: An instance of the `YOLO` class containing the model.
    Returns:
        None

    Raises:
        None
    """
    # Convert the video frame to an image
    image = frame.to_ndarray(format="bgr24")

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the choosen model
    res = model.predict(image, conf=conf)

    cv2.rectangle(image, (50, 100), (222, 222), (255, 0, 0), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")
    # image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the YOLOv8 model
    # res = model.predict(image, conf=conf)
    # logger.info("================================== RES ==================================")
    # logger.info(res)

    # # Plot the detected objects on the video frame
    # res_plotted = res[0].plot()
    
    # return res_plotted