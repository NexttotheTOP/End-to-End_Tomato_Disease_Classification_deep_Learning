from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image  # Model used to read in images from python 
import tensorflow as tf 
import requests # To make our http call 

app = FastAPI()

# Defining our chekpoint to use tf-serving and being able to choose and use between models 
origins = [
    "http://localhost",
    "http://localhost:3000",
]

endpoint = "http://localhost:8502/v1/models/Tomatoes_model:predict"
# Your chekpoint will be decided by how you run your docker image: port, ... 

# Model = tf.keras.models.load_model("/Users/wout_vp/Code/Tomato_Disease_Classification_Project_with_GDP_Deployment/Saved_Models/1")


Class_Names = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"] # Make sure the Class Names are the same as in your notebook 


# Function to make sure that our server is live and working 

@app.get("/")   
async def get_root():
    return "We are working babyy"

# Function that reads in our bytes and converts it into an image 

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  #  First reads in our bytes as a pillow image and then converts our image into a numpy array, which we will later be able to return as an image 
    return image

# Defining a post connecting where we will be able to upload our file which we will have predicted 
# Be aware that in order for the second post route to show op in the "localhost/docs" you will have to terminate the running process or close the terminal and restart this again after the changes are made
# I will be using Postman, if you're not familiar with it, you can watch some tutorials online and prectice using it, I strongly recommend you too

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())  # By using the "async" and "await" methods the application can while it is reading in files, contiue executing other tasks, the user interface will remain responsive instead of freezing.
    img_batch = np.expand_dims(image, 0) # Goes from [] to [[]] ; This is needed because the model.predict doesn't take in a single image, but only a batch; shape must be: [[256, 256, 3]]  ; check documenten API 
    
    json_data = {
        "instances": img_batch.tolist() # Expected/required format 
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])  # After our request we will receive a response in json which will contain our predictions 

    predicted_class = Class_Names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": confidence
    }


# The first way we can run uvicron is the usual way via terminal; "uvicorn main:app --reload"
# The second way below is more practical and this way we are also able to run it via different/other files

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)  

# This way we can run uvicorn by just running the file and we can make changes like port of host in our file itself

