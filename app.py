"""Streamlit skeleton for the WESAD stress project."""

import streamlit as st


st.set_page_config(
    page_title="WESAD Stress Detection",
    layout="wide",
)

st.title("Multimodal Stress-Pattern Detection")

st.warning(
    "This application is a research demonstration and not a medical diagnostic tool."
)

st.write(
    "The trained-model workflow is implemented in the notebooks. This app is a "
    "minimal placeholder for presenting future inference and model-comparison views."
)
