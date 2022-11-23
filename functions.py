import requests
import os
import time
from scipy.interpolate import griddata
import math
import numpy as np
from colour import Color
from PIL import Image
import shutil
from flask import request

# constrain the values of the colours within the limits
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

# map the value of the temperature to the colour
def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def generate_img(thermal_cam, max_values):
    COLORDEPTH = 1024

    # the points in the 8x8 grid
    points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]

    # the grid to store the 240x240 pixel values
    grid_x, grid_y = np.mgrid[0:7:240j, 0:7:240j]

    # listing the colours
    blue = Color("indigo")
    # the list of all colours from blue (indigo) to red with the depth as COLORDEPTH
    colors = list(blue.range_to(Color("red"), COLORDEPTH))
    # splitting into R, G, B
    colors = [(int(c.red * 255), int(c.green * 255), int(c.blue) * 255)
            for c in colors]
    for i in range(64):
        thermal_cam[i] = float(thermal_cam[i])
        MINTEMP = max_values[i]
        row = i / 8
        col = i % 8
        if row >= 1 and row <= 6 and col >= 1 and col <= 6:
            thermal_cam[i] = map_value(thermal_cam[i], MINTEMP, MINTEMP + 1.0, 0, COLORDEPTH - 1)
        else:
            thermal_cam[i] = map_value(thermal_cam[i], MINTEMP, MINTEMP + 0.5, 0, COLORDEPTH - 1)
    bicubic = griddata(points, thermal_cam, (grid_x, grid_y), method="cubic")   
    image_arr = np.zeros((240, 240, 3), dtype=np.uint8)
    for ix, row in enumerate(bicubic):
        for jx, pixel in enumerate(row):
            image_arr[ix][jx] = colors[constrain(int(pixel), 0, COLORDEPTH - 1)]
    image_render = Image.fromarray(image_arr)
    image_render.save("static/thermal_img.jpg")
    return "200"

def predict_count(model):
    import tensorflow as tf
    from tensorflow import keras
    img = tf.keras.utils.load_img('static/thermal_img.jpg', target_size=(240, 240))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_names = ['0', '1', '2', '3', '4']
    answer = class_names[np.argmax(score)], 100 * np.max(score)
    answer = list(answer)
    answer = answer[0]
    with open("static/prediction.txt", "w") as f:
        f.write(str(answer))
    return answer
    

