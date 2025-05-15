from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from http import HTTPStatus
import logging
import torch
import hydra
from hydra import compose
from torchvision import transforms
from PIL import Image
from dtu_mlops_team94.models.model import SeabedClassifier
import requests
import os
import wandb
import tempfile
from dtu_mlops_team94.utils.env import set_environment

logger = logging.getLogger(__name__)

app = FastAPI()

this_file_abs_dir = os.path.dirname(os.path.abspath(__file__))
conf_dir = os.path.join(this_file_abs_dir, "conf")
hydra.initialize_config_dir(config_dir=conf_dir)

config = compose(config_name="config")

# Set the environment variables (W&B API key)
set_environment(config)

def _find_ckpt_path(artifact_dir):
    # if the artifact_dir does not exist, create it
    if not os.path.exists(artifact_dir):
        logger.info(f"Artifacts folder not found, creating it at {artifact_dir}")
        os.mkdir(artifact_dir)

    # find the name of the .cktp file
    for file in os.listdir(artifact_dir):
        # file is a directory - look into the dir
        if os.path.isdir(os.path.join(artifact_dir, file)):
            return _find_ckpt_path(os.path.join(artifact_dir, file))

        if file.endswith(".ckpt"):
            return os.path.join(artifact_dir, file)

    return None


def intialize_model(always_load_from_wandb=False):
    # Load the model
    # model_path = "models/model_20240118-210715.ckpt"

    if not always_load_from_wandb:
        artif_dir = os.path.join(this_file_abs_dir, "../artifacts")
        ckpt_path = _find_ckpt_path(artif_dir)
        if ckpt_path is not None:
            print(f"Found existing ckpt, loading model from {ckpt_path}")
            model = SeabedClassifier.load_from_checkpoint(ckpt_path)
            return model

    model_path = ""
    # Download the model from W&B
    with wandb.init(project=config.project.wandb.project_name, job_type="inference") as run:
        artifact = run.use_artifact("jakubh/model-registry/SeabedClassifier:latest", type="model")
        artifact_dir = artifact.download()

        print(f"-- Artifact downloaded to: {artifact_dir}")

        # find the name of the .pth file
        for file in os.listdir(artifact_dir):
            if file.endswith(".ckpt"):
                model_path = os.path.join(artifact_dir, file)
                break
    print(f"-- loading model from {model_path}")
    model = SeabedClassifier.load_from_checkpoint(model_path)
    return model


model = intialize_model(always_load_from_wandb=True)
tempdir = tempfile.TemporaryDirectory()


def download_image(url, save_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
    else:
        raise HTTPException(status_code=response.status_code, detail="Failed to download image from URL.")


# @hydra.main(config_path="conf", config_name="config", version_base="1.1")
def process_image(image_content) -> None:
    # logger.info(OmegaConf.to_yaml(config))

    print(image_content)

    # Run prediction
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
        ]
    )

    image = Image.open(image_content).convert("RGB")  # Ensure image is in RGB mode
    content_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Send the tensor through the model
    model.eval()
    with torch.no_grad():
        y_pred = model(content_tensor)

    # Assuming your model outputs probabilities, not logits
    y_pred_probabilities = torch.nn.functional.softmax(y_pred[0], dim=0)
    y_pred_argmax = torch.argmax(y_pred_probabilities).item()
    label = {0: "Posidonia", 1: "Ripple 45", 2: "Rock", 3: "Sand", 4: "Silt", 5: "Ripple vertical"}

    # Convert tensor to NumPy array for JSON serialization
    y_pred_probabilities_numpy = y_pred_probabilities.cpu().numpy()

    # Return predictions as JSON
    return {
        "predictions": y_pred_argmax,
        "label": label[y_pred_argmax],
        "prediction_probability": y_pred_probabilities_numpy.tolist(),
    }


@app.post("/predict/")
# async def predict_image(file: UploadFile = File(None), image_url: str = Query(None, title="Image URL", description="URL of the image")):
async def predict_image(file: UploadFile = File(None)):
    try:
        if file:
            # Save the uploaded file to tempdir
            image_path = os.path.join(tempdir.name, file.filename)

            with open(image_path, "wb") as image_file:
                content = await file.read()
                image_file.write(content)
            result = process_image(
                image_path,
            )
        # elif image_url:
        #     # Download the image from the provided URL
        #     image_path = os.path.join(tempdir.name, "downloaded_image.jpg")
        #     download_image(image_url, image_path)
        #     result = process_image(
        #         image_path,
        #     )
        else:
            raise HTTPException(status_code=400, detail="Either 'file' or 'image_url' must be provided.")

        response = {
            # "input": file.filename if file else image_url,
            "input": file.filename,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
        }

        response.update(result)

        # Return predictions as JSON
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
