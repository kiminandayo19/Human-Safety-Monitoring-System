import os
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow

CONFIG = {
    "project_name": "ppe_detection_big",
    "pretrained_model_name": "yolo11m.pt",
    "rf_workspace": "arif-faishal",
    "rf_project_name": "helmet-detection-v3-9yama",
    "target_class": ['Hard Hat', 'No Hard Hat', 'Mask',
                     'No Mask', 'Safety Vest', 'No Safety Vest'],
    # NOTES: adjust based on the path of the dataset directory
    # this basepath is based on google colab code,
    "dataset_base_path": "/content/Helmet-Detection-v3-1",
    "out_dir_path": "ppe_detection",

    # Hyperparams
    "epochs": 50,
    "img_size": 640,
    "batch_size": 16
}


def train_model(yolo_model):
    os.makedirs(CONFIG["out_dir_path"], exist_ok=True)

    yolo_model.train(
        data=os.path.join(CONFIG["dataset_base_path"], "data.yaml"),
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["img_size"],
        batch=CONFIG["batch_size"],
        project=CONFIG["out_dir_path"],
        name=CONFIG["project_name"],
    )


def main():
    yolo_model = YOLO(CONFIG["pretrained_model_name"])

    rf = Roboflow(api_key=os.getenv("RF_API_KEY"))
    project = rf.workspace(CONFIG["rf_workspace"]).project(
        CONFIG["rf_project_name"])
    version = project.version(1)
    version.download("yolov11")

    train_model(yolo_model)


if __name__ == "__main__":
    load_dotenv()
    main()
