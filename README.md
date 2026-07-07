# LabhTeX Image Search

lIVE : https://labhtex-image-search-1.onrender.com/     

## About

This project is a simple image search system built using CLIP and FAISS.

The idea is simple:
- Convert images into embeddings using CLIP.
- Store those embeddings in FAISS.
- Upload a new image and find the top 5 similar images.

This project is built as part of the LabhTeX AI/ML Engineer Assessment.


## Technologies Used

- Python
- PyTorch
- CLIP
- FAISS
- Streamlit


## Project Structure

labhtex-image-search/

dataset/
- images/
- styles.csv

embeddings/

embed.py

search.py

app.py

requirements.txt

README.md

---

## How to Run

### Install packages

```bash
pip install -r requirements.txt
```

### Generate embeddings

```bash
python embed.py
```

### Create FAISS index

```bash
python search.py
```

### Run the application

```bash
streamlit run app.py
```

---

## Output

- Upload an image
- Get Top 5 similar images
- Display similarity scores

---

## Future Improvements

- Text-to-image search
- Better UI
- Metadata filtering
- Larger dataset support
