from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers import ViTImageProcessor
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io
import json
import requests
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from mangum import Mangum

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {
    "max_length": max_length,
    "num_beams": num_beams,
    "pad_token_id": tokenizer.pad_token_id
}

def imageToText(image_paths):
    images = []
    for image_path in image_paths:
        image = image_path
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        images.append(image)

    inputs = feature_extractor(images=images, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Generate attention_mask for pixel_values
    attention_mask = torch.ones(pixel_values.shape[:2], device=device)

    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        **gen_kwargs
    )


    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

# Read the description from a file
with open("api_description.md", "r") as file:
    api_description = file.read()

app = FastAPI(title="Image Caption Generator API",
    description=api_description,
    version="1.0.0",
    contact={
        "name": "Riddhi Halade",
        "email": "riddhihalade@gmail.com",
    })

handler = Mangum(app)

class ImageCaption(BaseModel):
    caption: str

@app.post("/predict/")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        result = imageToText([image])
        return ImageCaption(caption=result[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")
