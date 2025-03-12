import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
from torchvision.models import efficientnet_b0
import io
import torch.nn as nn
import gdown
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
app = Flask(__name__)

os.makedirs("model", exist_ok=True)
file_id = "199BJ-MmFWJX54MDmljCiwgOLR7Sr2L9l"
output = "model/weight.pth"

# print("Downloading the weights of the model from GDrive...")
# gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# print("Download finished!")

num_classes = 10

model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
# model.load_state_dict(torch.load("model/weight.pth"))
# model.to(device)
# model.eval()

# Load the model
model.load_state_dict(torch.load(output))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Adaptation pour EfficientNet
        transforms.ToTensor(),
    ]
)

mapping = {0: 'action', 1: 'animation', 2: 'comedy', 3: 'documentary', 4: 'drama', 5: 'fantasy', 6: 'horror', 7: 'romance', 8: 'science Fiction', 9: 'thriller'}

@app.route("/predict", methods=["POST"])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
    print(predicted[0])
    return jsonify({"prediction": mapping[int(predicted[0])]})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    # Get the image data from the request
    images_binary = request.files.getlist("images[]")

    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream)
        tensor = transform(img_pil)
        tensors.append(tensor)

    # Stack tensors to form a batch tensor
    batch_tensor = torch.stack(tensors, dim=0)

    # Make prediction
    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    return jsonify({"predictions": predictions.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
