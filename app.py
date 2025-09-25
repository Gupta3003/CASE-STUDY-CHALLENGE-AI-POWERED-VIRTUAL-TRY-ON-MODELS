# app.py 
import os
from flask import Flask, request, render_template, send_from_directory
import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image

from src.model_viton import Autoencoder
from src.utils import get_device, create_dir

UPLOAD_FOLDER = "static/uploads/"
OUT_FOLDER = "outputs/inference_samples/"
CHECKPOINT = "models/checkpoints/autoencoder_fast_epoch5.pth"
LATENT_DIM = 128

DEVICE = get_device()
create_dir(UPLOAD_FOLDER)
create_dir(OUT_FOLDER)

# Load model
model = Autoencoder(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

def denorm(x):
    return (x * 0.5) + 0.5

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = Image.open(filepath).convert("L")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            recon = model(img_tensor)
            recon_denorm = denorm(recon)

        out_path = os.path.join(OUT_FOLDER, "recon_" + file.filename)
        save_image(recon_denorm, out_path)

        return render_template("result.html", input_img=file.filename, output_img="recon_" + file.filename)

    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
