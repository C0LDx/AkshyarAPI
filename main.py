from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from model.segmenter import segment
from model.predict import predict

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
def ping():
    return {"connection": "OK"}

@app.post("/predict")
def get_prediction(image_data: bytes = Body(..., media_type="image/png")):
    try:
        img = Image.open(BytesIO(image_data)).convert('RGB')
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('segmented',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows() 
        segment(img)
        prediction = predict()
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")