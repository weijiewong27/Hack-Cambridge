# Import all required libaries
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.image as mpimg
import urllib

# Operating system dependencies
os.environ['KMP_DUPLICATE_LIB_OK']='True'
st.set_option('deprecation.showfileUploaderEncoding', False)

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the video detector", "Run the image detector", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("streamlit_app.py")) # change to st_newui.py" when uploaded to Github
    elif app_mode == "Run the video detector":
        readme_text.empty()
        run_video_detector()
    elif app_mode == "Run the image detector":
        readme_text.empty()
        run_image_detector()


# HELPER FUNCTIONS
def load_face_detector_model():
    """
    Loads the face detector model (EXTENSION: Train our own face detector model)
    """
    prototxt_path = os.path.sep.join(
        ["face_detector", "deploy.prototxt"])
    weight_path = os.path.sep.join(
        ['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])
    net = cv2.dnn.readNet(prototxt_path, weight_path)

    return net

# This will make the app stay performant
@st.cache(allow_output_mutation=True)
def load_mask_model():
    """
    Loads face mask detector model
    """
    mask_model = load_model("mask_detector_ewan.model")

    return mask_model

# Load both models
net = load_face_detector_model() # load face detector model
model = load_mask_model() # load mask detector model

# Create confidence level slider
confidence_selected = st.sidebar.slider(
    'Select a confidence range', 0.0, 0.1, 0.5, 0.1) # display button to adjust 'confidence' between 0 - 0.5

# Helper functions to load the image and loop over the detection (for video and image options)
def detect_mask_video(image):
    label='Starting...'
    startX, startY, endX, endY = 0,0,0,0
    color = 'g'
    
    # Pre-process image to fit input tensor of face detection model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB
    orig = image.copy() # get a copy of the image
    (h, w) = image.shape[:2] # get image height and weight
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # construct a blob from the image
                                 (104.0, 177.0, 123.0))

    # Set processed image as the input to the model and run forward pass to compute output
    net.setInput(blob)  # pass the blob through the detection, get region that differ in propertes, and the face region
    detection = net.forward() # run forward pass to compute output of layer

    for i in range(0, detection.shape[2]): # loop through the detection
        confidence = detection[0, 0, i, 2] # extract confidence value (something to do with how well the facial region is extracted)

        if confidence > confidence_selected: # if the confidence is greater than the selected confidence from the side bar

            # Generate face bounding box
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h]) # get x and y coordinate for the bounding box
            (startX, startY, endX, endY) = box.astype("int") 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY)) # ensure bounding box does not exceed image frame

            # Extract face
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # extract face ROI, convert from BGR to RGB
            face = cv2.resize(face, (128, 128))         # resize to input tensor size of mask model (128,128 - Ewan ; 224,224 - Crib)
            face = img_to_array(face)                      # convert resized face to an array
            face = preprocess_input(face)               # preprocess the array
            face = np.expand_dims(face, axis=0)            # expand array to 2D

             # Run extracted face through mask model and label prediction 
            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask on" if mask > withoutMask else "No Mask" 
            color = (0, 255, 0) if label == "Mask on" else (255, 0, 0) # bbox is Green if 'mask' else Red
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # add label probability 

            # Display label and bbox rectangle in output frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.20, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) 

        else:
            continue

        return image, label, startX, startY, endX, endY, color # return image and label


def detect_mask_image(image):
    # Pre-process image to fit input tensor of face detection model 
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)  # read the image from tempoary memory
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB
    orig = image.copy() # get a copy of the image
    (h, w) = image.shape[:2] # get image height and weight
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # construct a blob from the image
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)  # pass the blob through the detection, get region that differ in propertes, and the face region
    detection = net.forward() 

    for i in range(0, detection.shape[2]): # loop through the detection
        confidence = detection[0, 0, i, 2] # extract confidence value

        if confidence > confidence_selected: # if the confidence is greater than the selected confidence from the side bar

            # Generate face bounding box
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h]) # get x and y coordinate for the bounding box
            (startX, startY, endX, endY) = box.astype("int") 
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY)) # ensure bounding box does not exceed image frame

            # Extract face
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # extract face ROI, convert from BGR to RGB
            face = cv2.resize(face, (128, 128))         # resize to 224, 224
            face = img_to_array(face)                      # convert resized face to an array
            face = preprocess_input(face)               # preprocess the array
            face = np.expand_dims(face, axis=0)            # expand array to 2D

            # Run extracted face through mask model and label prediction 
            (mask, withoutMask) = model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask" # define label
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0) # bbox is Green if 'mask' else Blue
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # add label probability 

            # Display label and bbox rectangle in output frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) #display label and bbox rectangle in output frame

        return image, label # return image and label


def run_video_detector():
    st.title("Face Mask Detector Video App :mask:") # create App title
    run = st.checkbox('Run') # checkbox to run video 
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        image, label, startX, startY, endX, endY, color = detect_mask_video(frame) # call mask detection model
        FRAME_WINDOW.image(image) # NOTE: may need to crop this
        
    else:
        st.write('Stopped')


def run_image_detector():
    st.title("Face Mask Detector Image App :mask:") # create App title        
    image_file = st.file_uploader("Upload image", type=['jpeg', 'jpg', 'png']) # streamlit function to upload file
            
    if image_file is not None:  # Confirm that the image is not a 0 byte file
            st.sidebar.image(image_file, width=240) # then display a sidebar of the uploaded image
            
            if st.button("Process"): # Click button to run algorithm on input image
                image, label = detect_mask_image(image_file) # call mask detection model
                st.image(image, width=420) # display the uploaded image
                st.success('### ' +  label) # display label

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path # need to change URL
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if __name__ == "__main__":
    main()    