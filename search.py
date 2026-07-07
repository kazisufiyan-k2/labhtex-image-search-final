import pickle
import faiss
import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP Model

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

# Load FAISS Index

@st.cache_resource
def load_index():
    return faiss.read_index("embeddings/faiss.index")


# Load Image Paths

@st.cache_resource
def load_image_paths():
    with open("embeddings/image_paths.pkl", "rb") as f:
        return pickle.load(f)

model, processor = load_model()
index = load_index()
image_paths = load_image_paths()

# Search Function
def search_image(image, top_k=5):

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    features = features.cpu().numpy().astype("float32")

    scores, indices = index.search(features, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append(
            {
                "image": image_paths[idx],
                "score": round(float(score) * 100, 2)
            }
        )

    return results