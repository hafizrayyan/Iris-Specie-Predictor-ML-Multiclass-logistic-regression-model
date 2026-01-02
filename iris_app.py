import streamlit as st
import pickle
import numpy as np

with open("Iris_Specie_detector", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Iris Species Predictor",
    layout="centered"
)

st.markdown("""
<style>
/* Main page background */
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
}

/* Main page text color */
p, h1, h2, h3, h4, h5, h6, span, a, li {
    color: #FFFFFF;
}

/* Sidebar text color */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] a {
    color: black !important;
}

/* Button styling */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align:center; color: #FFFFFF; '> Iris Species Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Enter flower measurements to predict species</p>",
    unsafe_allow_html=True
)

st.sidebar.title("About Project")
st.sidebar.info(
    """
    **Iris Species Prediction App**

    This ML model predicts iris species using:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    Algorithm: **Machine Learning**
    Dataset: **Iris Dataset**
    """
)


st.divider()

# input section
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

st.divider()

# prediction
if st.button(" Predict Species", use_container_width=True):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    prediction = model.predict(input_data)[0]

    species_map = {
        0: " Iris Setosa",
        1: " Iris Versicolor",
        2: " Iris Virginica"
    }

    st.markdown(
    "<h3 style='color: white;'>Prediction Result</h3>",
    unsafe_allow_html=True
)
    
    st.success(f"**Predicted Species:** {species_map[prediction]}")

# Footer
st.markdown(
    "<hr><p style='text-align:center; font-size: 12px;'>Developed by Hafiz Rayyan Asif | Data Science & ML</p>",
    unsafe_allow_html=True
)
    
