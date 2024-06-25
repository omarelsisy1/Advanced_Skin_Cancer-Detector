from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn


# Define the App
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('/Users/Downloads/Best_Acc 95 %/EfficientNetB6_model.keras')
# Define the class names
class_names = ['df', 'bcc', 'vasc', 'akiec', 'mel', 'nv', 'bkl']


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))

        # Preprocess the image
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions)]
        probability = np.max(predictions)

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "probability": float(probability)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "_main_":
    uvicorn.run(app, port=8000)