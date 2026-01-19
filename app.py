from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/cnn_model.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

class_labels = sorted(os.listdir("dataset/train"))   # auto-load leaf disease names

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_path = "/static/uploads/" + file.filename

            img = image.load_img(filepath, target_size=(224,224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)
            class_index = np.argmax(pred)
            prediction = class_labels[class_index]

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
