import PySimpleGUI as sg
import cv2
from pathlib import Path
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras

# Load the model to classify sign
model = load_model('traffic_classifier.h5')
size = 300
image_to_classify_path = ""

# List of sign names
names = {1: 'Speed limit (20km/h)',
         2: 'Speed limit (30km/h)',
         3: 'Speed limit (50km/h)',
         4: 'Speed limit (60km/h)',
         5: 'Speed limit (70km/h)',
         6: 'Speed limit (80km/h)',
         7: 'End of speed limit (80km/h)',
         8: 'Speed limit (100km/h)',
         9: 'Speed limit (120km/h)',
         10: 'No passing',
         11: 'No passing vehicle over 3.5 tons',
         12: 'Right-of-way at intersection',
         13: 'Priority road',
         14: 'Yield',
         15: 'Stop',
         16: 'No vehicles',
         17: 'Vehicle over 3.5 tons prohibited',
         18: 'No entry',
         19: 'General caution',
         20: 'Dangerous curve left',
         21: 'Dangerous curve right',
         22: 'Double curve',
         23: 'Bumpy road',
         24: 'Slippery road',
         25: 'Road narrows on the right',
         26: 'Road work',
         27: 'Traffic signals',
         28: 'Pedestrians',
         29: 'Children crossing',
         30: 'Bicycles crossing',
         31: 'Beware of ice/snow',
         32: 'Wild animals crossing',
         33: 'End speed + passing limits',
         34: 'Turn right ahead',
         35: 'Turn left ahead',
         36: 'Ahead only',
         37: 'Go straight or right',
         38: 'Go straight or left',
         39: 'Keep right',
         40: 'Keep left',
         41: 'Roundabout mandatory',
         42: 'End of no passing',
         43: 'End no passing vehicle > 3.5 tons'
         }


# function
def classify():
    img = keras.preprocessing.image.load_img(image_to_classify_path, target_size=(30, 30))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = keras.applications.mobilenet.preprocess_input(img_array)
    pred = model.predict([img_array])
    window["SignName"].update(names[np.argmax(pred)])

    try:
        res, img_to_show = cv2.imencode(".png", cv2.imread(image_to_classify_path))
    except:
        window["Status"].update("Cannot identify image")
        return

    window["Result"].update(data=img_to_show.tobytes())


def read_file():
    global image_to_classify_path
    image_to_classify_path = sg.popup_get_file("", no_window=True)
    if image_to_classify_path == "":
        return

    window["Status"].update("Choose image")
    if not Path(image_to_classify_path).is_file():
        window["Status"].update("Image not found")
        return

    try:
        res, image = cv2.imencode(".png", cv2.imread(image_to_classify_path))
    except:
        window["Status"].update("Cannot identify image")
        return

    window["Image"].update(data=image.tobytes())


# gui
gui = [
    [sg.Text("Choose image", expand_x=True, key="Status")],
    [sg.Button("Browse"), sg.Button("Classify")],
    [sg.Image(size=(size, size), key="Image")]
]

# result gui
result_gui = [
    [sg.Text(key="SignName")],
    [sg.Image(size=(size, size), key="Result")]
]

layout = [
    [
        sg.Column(gui, vertical_alignment="center", justification="center"),
        sg.VSeperator(),
        sg.Column(result_gui, vertical_alignment="center", justification="center")
    ]
]

window = sg.Window("Project", layout)

# Event loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    elif event == "Browse":
        read_file()
    elif event == "Classify":
        classify()
window.close()
