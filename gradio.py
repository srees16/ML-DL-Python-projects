'''''''''''''''Gradio'''''''''''''''
#ref: https://www.youtube.com/watch?v=wruyZWre2sM, https://www.youtube.com/watch?v=O_3FINRtwhs

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

# ex: 3 https://www.youtube.com/watch?v=zoEJQr1VJ3Q
from PIL import Image

response = requests.get(url='https://git.io/JJkYN')
labels = response.text.split('\n')

mobile_net = tf.keras.applications.MobileNetV2()
inception_net = tf.keras.applications.InceptionV3()

def classify_mobile_net(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((224, 224))
    arr = np.array(img).reshape((-1, 224, 224, 3))
    arr = tf.keras.applications.mobilenet.preprocess_input(arr)
    prediction = mobile_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

def classify_inception_net(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = img.resize((299, 299))
    arr = np.array(img).reshape((-1, 299, 299, 3))
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    prediction = inception_net.predict(arr).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}

imagein = gr.inputs.Image()
label = gr.outputs.Label(num_top_classes = 3)

image_path = 'Desktop/Python_Projects/gradio/'
sample_images = [[image_path+'a.jpeg'], [image_path+'b.jpeg'], [image_path+'c.jpeg']]

gr.Interface(fn = [classify_mobile_net, classify_inception_net], inputs = imagein, outputs = label, capture_session=True, title='ImageNet vs Inception', examples = sample_images).launch(share = False)
