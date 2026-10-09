import streamlit as st
import requests

st.title("Chair Detection API")

uploaded = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded is not None:
    st.image(uploaded)

    # photo ko FastAPI ke /analyze ko bhejna
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    response = requests.post("http://127.0.0.1:8000/analyze", files=files)

    if response.status_code == 200:
        status = response.json()["status"]
        if status == "occupied":
            st.warning("Occupied")
        else:
            st.success("Empty")
    else:
        st.error(response.text)