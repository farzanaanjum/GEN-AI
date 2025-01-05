import base64
import io
import json
import streamlit as st
import boto3
from PIL import Image, ImageDraw, ImageFont

# Initialize boto3 client
boto3_bedrock = boto3.client('bedrock-runtime')

# Streamlit app
st.title("AI-Powered Advertisement Image Generator")

# User Inputs
prompt = st.text_input("Enter your product advertisement prompt:", "A stylish new smartphone with sleek design")

negative_prompts = [
    "poorly rendered",
    "poor background details",
    "poorly drawn elements",
    "disfigured features",
]

style_preset = st.selectbox("Select Style Preset:", ["photographic", "digital-art", "cinematic"])
clip_guidance_preset = st.selectbox("Select Clip Guidance Preset:", ["FAST_BLUE", "FAST_GREEN", "NONE", "SIMPLE", "SLOW", "SLOWER", "SLOWEST"])
sampler = st.selectbox("Select Sampler:", ["DDIM", "DDPM", "K_DPMPP_SDE", "K_DPMPP_2M", "K_DPMPP_2S_ANCESTRAL", "K_DPM_2", "K_DPM_2_ANCESTRAL", "K_EULER", "K_EULER_ANCESTRAL", "K_HEUN", "K_LMS"])
width = st.slider("Select Width:", min_value=256, max_value=2048, value=768, step=32)
height = st.slider("Select Height:", min_value=256, max_value=2048, value=1024, step=32)

# Add text overlay options
overlay_text = st.text_input("Add text overlay (optional):", "")
text_color = st.color_picker("Select text color:", "#FFFFFF")
text_size = st.slider("Select text size:", min_value=20, max_value=100, value=40)

# Prepare the request payload
request = json.dumps({
    "text_prompts": (
        [{"text": prompt, "weight": 1.0}]
        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
    ),
    "cfg_scale": 5,
    "seed": 42,
    "steps": 60,
    "style_preset": style_preset,
    "clip_guidance_preset": clip_guidance_preset,
    "sampler": sampler,
    "width": width,
    "height": height,
})
modelId = "stability.stable-diffusion-xl-v1"

if st.button("Generate Image"):
    # Call the Bedrock model
    response = boto3_bedrock.invoke_model(body=request, modelId=modelId)
    response_body = json.loads(response.get("body").read())

    # Extract the base64 image string
    base_64_img_str = response_body["artifacts"][0].get("base64")

    # Convert base64 string to an image
    image_bytes = io.BytesIO(base64.b64decode(base_64_img_str))
    image = Image.open(image_bytes)
    
    # Apply text overlay if provided
    if overlay_text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", text_size)  # Use a font file available in your environment
        text_width, text_height = draw.textsize(overlay_text, font=font)
        width, height = image.size
        position = (width // 2 - text_width // 2, height - text_height - 20)  # Position text at the bottom center
        draw.text(position, overlay_text, fill=text_color, font=font)

    # Display the image with options to download
    st.image(image, caption='Generated Advertisement Image', use_column_width=True)
    st.download_button(label="Download Image", data=image_bytes.getvalue(), file_name="advertisement_image.png", mime="image/png")

