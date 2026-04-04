import base64
import io
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the PyTorch YOLOv5 model natively from ultralytics hub
print("Downloading/Loading YOLOv5n model from torch hub...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4  # Translating CLASS_THRESHOLD = 0.4 from the JS worker

class PredictRequest(BaseModel):
    image: str

@app.post("/predict")
def predict_api(req: PredictRequest):
    # Decode the Base64 image sent from the canvas
    img_data = base64.b64decode(req.image.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    
    # Run YOLOv5 Inference
    results = model(img)
    
    # Extract pandas dataframe representation
    df = results.pandas().xyxy[0]
    
    # Filter only for the 'kite' class, as we did in JS: if (label !== 'kite') continue
    kites = df[df['name'] == 'kite']
    
    predictions = []
    for _, row in kites.iterrows():
        x1, y1 = row['xmin'], row['ymin']
        x2, y2 = row['xmax'], row['ymax']
        
        box_width = x2 - x1
        box_height = y2 - y1
        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2
        
        predictions.append({
            "type": "prediction",
            "x": center_x,
            "y": center_y,
            "score": f"{row['confidence'] * 100:.2f}"
        })
        
    return {"predictions": predictions}
