from flask import Flask, render_template, request, redirect
import functions
from PIL import Image
import shutil
import tensorflow as tf
max_values = [24.25] * 64
thermal_cam = [24.25] * 64
model = tf.keras.models.load_model("static/trained_cnn2")
count = [0]
app = Flask(__name__)

@app.route("/status", methods=['GET', 'POST'])
def index():
    return render_template("data.html")

@app.route("/calibrate")
def calibrate():
    base = count[0]
    for i in range(64):
        max_values[i] = 0

    while count[0] - base < 10:
        for i in range(64):
            max_values[i] = max(max_values[i], thermal_cam[i])
    print("calibrated successfully!!")
    return "200"

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/obtain", methods=["GET", "POST"])
def obtain_count():
    count[0] = count[0] + 1
    for i in range(10):
        thermal_cam[i] = float(request.form.get("val0" + str(i)))
    for i in range(10, 64):
        thermal_cam[i] = float(request.form.get("val" + str(i)))
    functions.generate_img(thermal_cam, max_values)
    value = functions.predict_count(model)
    shutil.copy("static/" + str(value) + ".png", "static/curr.png")
    return "200"    









