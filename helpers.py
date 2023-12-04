import logging
import os
from requests import get
from torch import ge

from ultralytics import YOLO
import av
import cv2

import settings
import helpers

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

def play_webcam(frame: av.VideoFrame, conf: float, models) -> av.VideoFrame:
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
    image = cv2.resize(image, (640, int(640*(9/16))))

    # Predict the objects in the image using the choosen model
    results: YOLO = models["DETECTION"].predict(image, conf=conf)

    results_pose = helpers.load_model(settings.POSE_MODEL).predict(image, conf=conf)

    for r in results:
        im_array = r.plot()
        im_array = results_pose.plot(im_array, boxes=False)
        

    # cv2.rectangle(image, (50, 100), (222, 222), (255, 0, 0), 2)
    return av.VideoFrame.from_ndarray(im_array, format="bgr24")

def read_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    bounding_boxes = []
    for line in lines:
        parts = line.strip().split(' ')
        object_class = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        bounding_boxes.append((object_class, x, y, width, height))
    
    return bounding_boxes

def read_database(database_path):
    data = []
    
    for file_name in os.listdir(database_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(database_path, file_name)
            label_path = os.path.join(database_path, 'labels', f'label{file_name.split(".")[0][3:]}.txt')

            if os.path.exists(label_path):
                bounding_boxes = read_labels(label_path)
                data.append({'image_path': image_path, 'bounding_boxes': bounding_boxes})

    return data

def get_pose_estimation_data():
    pass

def get_hand_regions():
    pass

def get_object_detection_data():
    pass

def compare_hand_regions():
    pass

def start_evaluation(with_pose = False):
    """
    Starts the evaluation process.

    Parameters:
        None

    Returns:
        None

    Raises:
        None
    """

    # Load the model
    try:
        model = load_model(settings.DETECTION_MODEL)
        logger.info("Detection Model Loaded Successfully")

        if(with_pose):
            model_pose = load_model(settings.POSE_MODEL)
            logger.info("Pose Estimation Model Loaded Successfully")
    except Exception as ex:
        logger.error("Unable to load model. Check the specified path")
        logger.error(ex)

    # Read the dataset
    dataset = read_database(settings.DATASET_DIR)

    for entry in dataset:
        print(f"Image Path: {entry['image_path']}")
        for obj_class, x, y, width, height in entry['bounding_boxes']:
            print(f"Object Class: {obj_class}, Bounding Box: ({x}, {y}, {width}, {height})")
        print("\n")
    
    if with_pose:
        get_pose_estimation_data()

    get_hand_regions()

    get_object_detection_data()

    compare_hand_regions()