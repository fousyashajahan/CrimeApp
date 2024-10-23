import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Load your model and model columns
model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Add custom CSS for styling
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background-color: #f0f4f8;
        font-family: 'Arial', sans-serif;
        height: 100vh;
        margin: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 2rem;
        overflow-y: auto;
    }
    .container {
        text-align: center;
        width: 100%;
        max-width: 400px;
        background-color: #dad5d5;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1a202c;
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    label {
        display: block;
        font-size: 1rem;
        color: #333;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    input[type="time"], select {
        width: 100%;
        padding: 0.5rem;
        font-size: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box;
    }
    button {
        background-color: #007bff;
        color: white;
        font-size: 1rem;
        border: none;
        border-radius: 5px;
        padding: 0.75rem;
        cursor: pointer;
        width: 100%;
    }
    button:hover {
        background-color: #0056b3;
    }
    img {
        display: block;
        margin: 20px auto;
        max-width: 80%;
        height: auto;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Call the function to include CSS
add_custom_css()

# Streamlit App Title
st.title('Crime Predictor')

# City selection dropdown
city = st.selectbox("Select a City:", 
                    ['Ahmedabad', 'Chennai', 'Delhi', 'Mumbai', 'Pune', 'Bangalore', 
                     'Visakhapatnam', 'Surat', 'Ludhiana', 'Kolkata', 'Lucknow'])

# Time input field (24-hour format)
time_input = st.time_input('Select Hour of the Day (24-Hour Format):')

# Predict button
if st.button('Predict'):
    try:
        # Convert the time to hour
        hour = time_input.hour

        # Prepare the input data for the model
        input_data = pd.DataFrame([[hour]], columns=['Hour'])

        # Encode the city input into the model columns
        city_input = city.strip().lower()
        for city_col in model_columns:
            if city_col.startswith('City_'):
                city_name = city_col.split('City_')[1].strip().lower()
                input_data[city_col] = 1 if city_name == city_input else 0

        # Ensure the input_data matches the model columns
        input_data_encoded = input_data.reindex(columns=model_columns, fill_value=0)

        # Perform prediction
        prediction_probs = model.predict_proba(input_data_encoded)[0]

        # Define crime labels
        crime_labels = ['Theft', 'Assault', 'Robbery', 'Burglary', 'Other']

        # Zip crime labels and prediction probabilities
        crime_predictions = zip(crime_labels, prediction_probs)

        # Generate the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(crime_labels, prediction_probs * 100, color='skyblue')
        plt.xlabel('Crime Type')
        plt.ylabel('Probability (%)')
        plt.title(f'Crime Occurrence Probability in {city} at {hour}:00')

        # Convert plot to PNG image to display in Streamlit
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        # Display the crime predictions
        st.subheader(f'Crime Occurrence Probability in {city} at {hour}:00')
        st.image(f"data:image/png;base64,{graph_url}")

        # Display probabilities by crime type
        st.subheader('Probabilities by Crime Type:')
        for crime_type, prob in crime_predictions:
            st.write(f"{crime_type}: {round(prob * 100, 2)}%")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
