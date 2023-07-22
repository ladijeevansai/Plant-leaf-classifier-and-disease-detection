from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

from pest_display import pest_dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Leaf_MODEL = tf.keras.models.load_model("C:/Users/ladij/OneDrive/Documents/Tarp Project/Training2/models_leaf_3/1")
Potato_MODEL = tf.keras.models.load_model("C:/Users/ladij/OneDrive/Documents/Tarp Project/Training2/models_potato/1")
Tomato_MODEL = tf.keras.models.load_model("C:/Users/ladij/OneDrive/Documents/Tarp Project/Training2/models_tomato/1")
Pepper_MODEL = tf.keras.models.load_model("C:/Users/ladij/OneDrive/Documents/Tarp Project/Training2/models_pepper/1")


leaf_CN = ["Pepper__bell","Potato","Tomato"]
potato_CN = ["Early Blight", "Late Blight", "Healthy"]
tomato_CN = ["Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_Late_blight","Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
    "Tomato_Tomato_YellowLeafCurl_Virus","Tomato_Tomato_mosaic_virus","Tomato_healthy"]

pepper_CN = ["Pepper_bell_Bacterial_spot", "Pepperbell__healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = Leaf_MODEL.predict(img_batch)

    predicted_class = leaf_CN[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])


    if predicted_class=='Potato':
        pred_potato = Potato_MODEL.predict(img_batch)
        pred_class_potato = potato_CN[np.argmax(pred_potato[0])]
        confidence_potato = np.max(pred_potato[0])

        
        x = pred_class_potato,
        y =  predicted_class,
        z = confidence_potato
        pests = pest_dict.get(pred_class_potato, {}).get("Pesticides", [])
        chems = pest_dict.get(pred_class_potato, {}).get("Chemicals", [])
        

    elif predicted_class=='Pepper__bell':
        pred_pepper = Pepper_MODEL.predict(img_batch)
        pred_class_pepper = pepper_CN[np.argmax(pred_pepper[0])]
        confidence_pepper = np.max(pred_pepper[0])

        x = pred_class_pepper,
        y = predicted_class,
        z = confidence_pepper
        pests = pest_dict.get(pred_class_pepper, {}).get("Pesticides", [])
        chems = pest_dict.get(pred_class_pepper, {}).get("Chemicals", [])


    elif predicted_class=='Tomato':
        pred_tomato = Tomato_MODEL.predict(img_batch)
        pred_class_tomato = tomato_CN[np.argmax(pred_tomato[0])]
        confidence_tomato = np.max(pred_tomato[0])

        x =  pred_class_tomato,
        y =  predicted_class,
        z = confidence_tomato
        pests = pest_dict.get(pred_class_tomato, {}).get("Pesticides", [])
        chems = pest_dict.get(pred_class_tomato ,{}).get("Chemicals", [])
      

    return JSONResponse({
        'leaf_name': y,
        'disease_name': x,
        'confidence': float(z),
        'pesticides': pests,
        'chemicals' : chems
    })
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)