import os
import numpy as np
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Load model & labels ──
model = load_model("models/model.keras")

with open("labels.json") as f:
    class_indices = json.load(f)

labels = {v: k for k, v in class_indices.items()}

# ── Upload config ──
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 224


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_id = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    raw_label = labels[class_id]
    # Clean underscores for display
    label = raw_label.replace("__", " ").replace("_", " ").strip()

    return label, round(confidence, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result="No file uploaded", confidence=None, img_path=None)

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", result="No file selected", confidence=None, img_path=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence = predict(filepath)
            result = label
            # Pass as a URL-friendly static path
            img_path = filepath.replace("\\", "/")

    return render_template("index.html", result=result, confidence=confidence, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=False)