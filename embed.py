import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

# PATHS

IMAGE_FOLDER = "dataset/images"
OUTPUT_FOLDER = "embeddings"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using Device: {device}")


# LOAD MODEL

print("Loading CLIP Model...")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

print("CLIP Loaded Successfully!")

# LOAD IMAGES

image_paths = []

for file in os.listdir(IMAGE_FOLDER):

    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_paths.append(
            os.path.join(IMAGE_FOLDER, file)
        )

image_paths.sort()

print(f"Total Images Found: {len(image_paths)}")

# CREATE EMBEDDINGS
embeddings = []

for path in tqdm(image_paths):

    try:

        image = Image.open(path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            feature = model.get_image_features(**inputs)

        feature = feature / feature.norm(dim=-1, keepdim=True)

        embeddings.append(
            feature.cpu().numpy()[0]
        )

    except Exception as e:

        print(path, e)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print("Embedding Shape:", embeddings.shape)

# SAVE EMBEDDINGS

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "embeddings.npy"
    ),
    embeddings
)

with open(
    os.path.join(
        OUTPUT_FOLDER,
        "image_paths.pkl"
    ),
    "wb"
) as f:

    pickle.dump(image_paths, f)

# CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    os.path.join(
        OUTPUT_FOLDER,
        "faiss.index"
    )
)

print("\n==============================")
print("Embeddings Saved Successfully!")
print("Images Indexed:", len(image_paths))
print("Output Folder:", OUTPUT_FOLDER)