import os
import random
import shutil
import pandas as pd

# Paths
source_images = "dataset/images"
source_csv = "dataset/styles.csv"

new_dataset = "dataset"
new_images = os.path.join(new_dataset, "images")

os.makedirs(new_images, exist_ok=True)

# Read all jpg images
all_images = [f for f in os.listdir(source_images) if f.lower().endswith(".jpg")]

# Randomly select 2000 images
selected = random.sample(all_images, 2000)

# Copy images
for img in selected:
    shutil.copy2(
        os.path.join(source_images, img),
        os.path.join(new_images, img)
    )

# Read styles.csv
df = pd.read_csv(source_csv, on_bad_lines="skip")

# Get selected IDs
ids = [int(os.path.splitext(i)[0]) for i in selected]

# Filter CSV
df = df[df["id"].isin(ids)]

# Save new CSV
df.to_csv(os.path.join(new_dataset, "styles.csv"), index=False)

print("=" * 50)
print("Done!")
print("Images selected:", len(selected))
print("CSV rows:", len(df))
print("Saved in:", new_dataset)