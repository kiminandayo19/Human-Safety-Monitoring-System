import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()


def download_model():
    models_dir = Path(__file__).parents[2] / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = hf_hub_download(
        repo_id="arf-dev/yolo-ppe",
        filename="yolo_50.pt",
        token=os.environ["HF_TOKEN"],
        local_dir=models_dir,
    )

    model_path_2 = hf_hub_download(
        repo_id="arf-dev/yolo-ppe",
        filename="yolo_person_tracker.pt",
        token=os.environ["HF_TOKEN"],
        local_dir=models_dir
    )

    return model_path, model_path_2
