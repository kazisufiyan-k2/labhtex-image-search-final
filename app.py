import streamlit as st
from PIL import Image
from search import search_image

# PAGE CONFIG

st.set_page_config(
    page_title="AI Fashion Image Search",
    layout="wide"
)

# TITLE
st.title("AI Fashion Image Search")
st.markdown(
    "Upload a fashion image and find the **Top-5 most similar products** using **CLIP + FAISS**."
)

st.divider()

# SIDEBAR

st.sidebar.header("Project Information")

st.sidebar.write("**Model:** CLIP (ViT-B/32)")
st.sidebar.write("**Similarity Search:** FAISS")
st.sidebar.write("**Dataset:** Fashion Images")
st.sidebar.write("**Results:** Top-5 Similar Images")


# FILE UPLOAD

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# SEARCH

if uploaded_file is not None:

    query_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 3])

    with col1:

        st.subheader("Query Image")
        st.image(query_image, width=250)

    with col2:

        st.subheader("Searching...")

        with st.spinner("Finding similar images..."):

            results = search_image(query_image, top_k=5)

        st.success("Search Completed!")

    st.divider()

    st.subheader("Top 5 Similar Images")

    cols = st.columns(5)

    for col, item in zip(cols, results):

        with col:

            image = Image.open(item["image"])

            st.image(
                image,
                width="stretch"
            )

            st.markdown(
                f"**Similarity:** {item['score']:.2f}%"
            )

else:

    st.info("Please upload an image to begin searching.")