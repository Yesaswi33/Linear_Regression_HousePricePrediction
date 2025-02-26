import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

# --- Custom CSS for 3D Effects ---
st.markdown(
    """
    <style>
    *{
        font-family: 'Times New Roman', Times, serif;

    }
    .main {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        box-shadow:  7px 7px 15px #bebebe,
                     -7px -7px 15px #ffffff;
    }
    
    /* 3D Button effect */
    .stButton>button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        box-shadow:  4px 4px 8px #b1b1b1,
                     -4px -4px 8px #ffffff;
        transition: transform 0.1s;
    }
    .stButton>button:active {
        transform: translateY(4px);
        box-shadow:  2px 2px 4px #b1b1b1,
                     -2px -2px 4px #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Wrap content in a container to apply custom styles
st.markdown('<div class="main">', unsafe_allow_html=True)

# Streamlit App Title and Description
st.title("🏡 House Price Prediction App")
st.write("Enter the details below to predict the house price:")

# User input fields
income = st.number_input("Avg. Area Income ($)", min_value=0)
house_age = st.number_input("Avg. Area House Age (years)", min_value=0.0, format="%.2f")
rooms = st.number_input("Avg. Area Number of Rooms", min_value=0.0, format="%.2f")
bedrooms = st.number_input("Avg. Area Number of Bedrooms", min_value=0.0, format="%.2f")
population = st.number_input("Area Population", min_value=0)

# Predict button
if st.button("Predict House Price"):
    features = np.array([[income, house_age, rooms, bedrooms, population]])
    prediction = model.predict(features)[0]
    
    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")

# Close the container div
st.markdown('</div>', unsafe_allow_html=True)
