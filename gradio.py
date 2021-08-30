'''''''''''''''Gradio'''''''''''''''
#ref: https://www.youtube.com/watch?v=wruyZWre2sM

import gradio as gr
import tensorflow as tf
import requests
import numpy as np

# ex: 1
def greet(user):
    return 'welcome ' + user

iface = gr.Interface(fn = greet, inputs = 'text', outputs='text')
iface.launch(share=False) # True to create public link

# ex: 2
inception = tf.keras.applications.InceptionV3() # load model
response = requests.get(url='https://git.io/JJkYN') # download labels for Imagenet
labels = response.text.split('\n')

def classify_images(input_):
    input_ = input_.reshape((-1, 299, 299, 3))
    input_ = tf.keras.applications.inception_v3.preprocess_input(input_)
    prediction = inception.predict(input_).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

image = gr.inputs.Image(shape=(299, 299))
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn = classify_images, inputs = image, outputs = label, capture_session=True).launch(share = True)