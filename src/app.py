import io
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from texture_synthesis import TextureSynthesis
from PyQt5.QtCore import QBuffer, QIODevice
from utils import pil_to_qimage, resize_to_target
from PIL import Image

app = FastAPI()
texture_synthesis = TextureSynthesis()


@app.post("/generate-textures/")
async def generate_textures(
    user_image: UploadFile = File(...),
    user_mask: UploadFile = File(...),
    prompt: str = Form(...),
):
    # Convert uploaded files to PIL images
    user_image = Image.open(io.BytesIO(await user_image.read())).convert("RGB")
    user_mask = Image.open(io.BytesIO(await user_mask.read())).convert("L")

    # Resize images to match processing requirements
    user_image = resize_to_target(user_image, target_size=(1024, 1024))
    user_mask = resize_to_target(user_mask, target_size=(1024, 1024))

    # Generate textures
    images = texture_synthesis.generate_images(
        sample_steps=11,
        guidance_scale=8,
        # maps={"albedo", "rough", "normal", "height", "specular", "ambientocl", "metal"},
        maps={"albedo", "normal"},
        user_image=user_image,
        user_mask=user_mask,
        prompt=prompt,
    )

    # Convert images to byte format for response
    response_images = {}
    for name, img in images.items():
        qimage = pil_to_qimage(img)
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        qimage.save(buffer, "PNG")
        img_bytes = buffer.data().data()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        response_images[name] = img_base64

    return JSONResponse(content=response_images)
