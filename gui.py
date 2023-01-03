import PySimpleGUI as sg
import cv2
from pathlib import Path
from tkinter import *
from PIL import Image
from keras.models import load_model
import numpy

# Load the model to classify sign
model = load_model('traffic_classifier.h5')
size = 300

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
         11: 'No passing veh over 3.5 tons',
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
         22: 'Double curve'
         }


# function
def classify():
    print(window["Image"].image)
    # image = window["Image"].data
    # image = numpy.array(image)
    # pred = model.predict_step(image)[0]
    # sign = names[pred + 1]
    # window["Result"].draw_text(sign)


def readFile():
    path = sg.popup_get_file("", no_window=True)
    if path == "":
        return

    window["Status"].update("Choose image")
    if not Path(path).is_file():
        window["Status"].update("Image not found")
        return

    try:
        res, image = cv2.imencode(".png", cv2.imread(path))
    except:
        window["Status"].update("Cannot identify image")
        return

    window["Image"].update(data=image.tobytes())
    pred = model.predict_step(image)[0]


# gui
gui = [
    [sg.Text("Choose image", expand_x=True, key="Status")],
    [sg.Button("Browse"), sg.Button("Classify")],
    [sg.Image(size=(size, size), key="Image")]
]

layout = [
    [
        sg.Column(gui, vertical_alignment="center", justification="center"),
        sg.VSeperator(),
        sg.Text(key="SignName"),
        sg.Image(size=(size, size), key="Result")
    ]
]

window = sg.Window("Project", layout)

# Event loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    elif event == "Browse":
        readFile()
    elif event == "Classify":
        classify()
window.close()
