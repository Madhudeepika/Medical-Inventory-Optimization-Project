import pandas as pd
import numpy as np
import streamlit as st
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained SARIMAX model
sarimax_model_path = (r"C:\Users\Lenovo\my project\sarimax_model.pkl")
sarimax_model = joblib.load(sarimax_model_path)

# Set page color and background image
page_bg_img = '''
<style>
body {
background-image: url(r"A:\Downloads\drugimage.jpg");
background-size: cover;
}
</style>
'''

# Apply the styling
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Med_Inventory Optimization")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read the dataset
    data = pd.read_csv(uploaded_file)  # You might need to adjust this based on the file format

    # Make predictions using the SARIMAX model
    predictions = sarimax_model.predict(start=len(data), end=len(data) + 15 - 1, dynamic=False)

    # Display the predictions
    st.subheader("Forecast for the Next 15 Weeks:")
    st.write(predictions)

    # Plot a histogram of the forecasted quantities
    st.subheader("Histogram of Forecasted Quantities")
    plt.figure(figsize=(10, 5))
    sns.histplot(predictions, bins=20, kde=True)
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    st.pyplot(plt)


# Note: Adjust the preprocess_data function based on how you preprocessed the data during model training
#r"C:\Users\Lenovo\my project\sarimax_model.pkl")
 #   r"A:\Downloads\drugimage.jpg"