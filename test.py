from ultralytics import YOLO
import settings


model = YOLO(settings.POSE_MODEL)

results = model.predict(source="1", show=True)