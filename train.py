# Run this code in Google Colab with a T4 GPU.
# First run this in Colab:  !pip install roboflow ultralytics

from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="hMvrwQ8OhfDwLXrpw6pE")  # put your own key here, never share it
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(13).download("yolov8")

model = YOLO("yolov8n.pt")
model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    fraction=0.02,
    name="plate_small",
)
