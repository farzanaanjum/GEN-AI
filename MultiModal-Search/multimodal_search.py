import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import base64
from io import BytesIO
import boto3
from scipy.spatial.distance import cdist

# Initialize Bedrock client (AWS credentials required)
bedrock_client = boto3.client(
    "bedrock-runtime",
    "us-west-2",   # Adjust region as necessary
)

# Function to generate images
def titan_image(payload: dict, num_image: int = 1, cfg: float = 10.0, seed: int = 2024) -> list:
    body = json.dumps(
        {
            **payload,
            "imageGenerationConfig": {
                "numberOfImages": num_image,
                "quality": "premium",
                "height": 1024,
                "width": 1024,
                "cfgScale": cfg,
                "seed": seed,
            }
        }
    )

    response = bedrock_client.invoke_model(
        body=body,
        modelId="amazon.titan-image-generator-v1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    images = [
        Image.open(BytesIO(base64.b64decode(base64_image))) for base64_image in response_body.get("images")
    ]

    return images


# Function to get multimodal embedding
def titan_multimodal_embedding(image_path: str = None, description: str = None, dimension: int = 1024) -> dict:
    payload_body = {}
    embedding_config = {"embeddingConfig": {"outputEmbeddingLength": dimension}}

    if image_path:
        with open(image_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
        payload_body["inputImage"] = input_image
    if description:
        payload_body["inputText"] = description

    assert payload_body, "please provide either an image and/or a text description"

    response = bedrock_client.invoke_model(
        body=json.dumps({**payload_body, **embedding_config}),
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json",
    )

    return json.loads(response.get("body").read())["embedding"]


# Function to perform search using cosine distance
def search(query_emb: np.array, indexes: np.array, top_k: int = 1):
    dist = cdist(query_emb, indexes, metric="cosine")
    return dist.argsort(axis=-1)[0, :top_k], np.sort(dist, axis=-1)[0, :top_k]


# Streamlit App Interface
st.title("E-commerce Product Generator & Search")

# User input section
st.header("Generate Product Images")

product_options = [
    "T-shirt",
    "Jeans",
    "Sneakers",
    "Backpack",
    "Smartwatch",
    "Coffee maker",
    "Yoga mat",
]

selected_product = st.selectbox("Choose a product", product_options)
description = st.text_input("Enter additional description (optional)", "")

if st.button("Generate Image"):
    prompt = f"{selected_product}: {description}" if description else selected_product
    images = titan_image({"taskType": "TEXT_IMAGE", "textToImageParams": {"text": prompt}})
    
    for img in images:
        st.image(img, caption=selected_product, use_column_width=True)
        
        # Save the generated image
        image_path = f"generated_{selected_product}.png"
        img.save(image_path, format="png")
        
        # Get embedding
        embedding = titan_multimodal_embedding(image_path=image_path)
        st.session_state['embeddings'] = st.session_state.get('embeddings', []) + [embedding]
        st.session_state['titles'] = st.session_state.get('titles', []) + [image_path]
        st.success(f"Image generated and saved as {image_path}")



st.header("Search for Similar Products")

search_query = st.text_input("Enter search query")
top_k = st.slider("Number of results to show", 1, 5, 1)

if st.button("Search"):
    query_emb = titan_multimodal_embedding(description=search_query)
    query_emb = np.array(query_emb).reshape(1, -1)  # Ensure it's 2D
    embeddings = np.array(st.session_state.get('embeddings', []))

    if embeddings.size == 0:
        st.error("No images available to search. Please generate images first.")
    else:
        idx_returned, dist = search(query_emb, embeddings, top_k=top_k)
        
        # Print shapes for debugging
        st.write(f"Indices returned: {idx_returned}")
        st.write(f"Distances shape: {dist.shape}")
        
        for i, idx in enumerate(idx_returned):
            image_path = st.session_state['titles'][idx]
            st.image(Image.open(image_path), caption=f"Result: {image_path}", use_column_width=True)
            st.write(f"Cosine distance: {dist[i]:.4f}")
