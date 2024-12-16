import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px

# Load car dataset
car_df = pd.read_csv("merged_data.csv")

# Load the model and scaler
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load('scaler.pkl')

# Initialize LabelEncoders for categorical features
categorical_columns = [
    'bodytype', 'fueltype', 'transmission', 'DriveType', 'Insurance', 'oem', 'city'
]
# #label_encoders = {col: LabelEncoder().fit(car_df[col]) for col in categorical_columns}
# label_encoder_bodytype = joblib.load("labelencoded.pkl")
# onehotencoder = joblib.load("onehotencoded.pkl")

# Function to predict the resale price
def predict_resale_price(m_bodytype, m_seats, m_km, m_modelYear, m_ownerNo, 
                        m_Engine, m_gear, m_mileage, m_fuel_type,
                        m_transmission, m_Insurance, m_oem, m_drivetype, m_city):
    # Prepare numerical features
    num_features = np.array([
        int(m_seats),
        int(m_km),
        int(m_modelYear),
        int(m_ownerNo),
        int(m_Engine),
        int(m_gear),
        float(m_mileage)
    ]).reshape(1, -1)
    
    # Scale numerical features
    scaled_num_features = scaler.transform(num_features)
    
    # Prepare categorical features for one-hot encoding
    categorical_data = pd.DataFrame({
        'bodytype': [m_bodytype],
        'fueltype': [m_fuel_type],
        'transmission': [m_transmission],
        'Drive_Type': [m_drivetype],
        'Insurance_Validity': [m_Insurance],
        'oem': [m_oem],
        'City': [m_city]
    })
    
    # One-hot encode categorical features
    encoded_cats = pd.get_dummies(categorical_data, columns=categorical_data.columns)
    
    # Ensure all columns from training are present
    for col in model.feature_names_in_:
        if col not in encoded_cats.columns:
            encoded_cats[col] = 0
            
    # Reorder columns to match training data
    encoded_cats = encoded_cats[model.feature_names_in_[7:]]  # Skip numerical features
    
    # Combine numerical and categorical features
    final_features = np.hstack((scaled_num_features, encoded_cats))
    
    # Make prediction
    prediction = model.predict(final_features)
    return prediction[0]


# Streamlit Page Configuration

# Title

import streamlit as st
# Move all other imports here at the top

# Place set_page_config as the very first Streamlit command
st.set_page_config(
    layout="wide",
    page_icon=":material/directions_bus:",
    page_title="CarPrediction Project",
    initial_sidebar_state="expanded"
)

# Rest of your Streamlit app code goes here

st.markdown(
    """
    <style>
    .logo-container {
        position: relative;
        float: right;
        margin-top: 80px;
        margin-right: 20px;
        z-index: 999;
    }
    .logo-image {
        width: 150px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    <div class="logo-container">
        <img class="logo-image" src="https://stimg2.cardekho.com/images/carNewsimages/userimages/650X420/30183/1672738680556/GeneralNew.jpg">
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    f"""
    <style>
    .stApp {{
        background-size: cover; /* Ensures the image covers the entire container */
        background-position: center; /* Centers the image */
        background-repeat: no-repeat; /* Prevents the image from repeating */
        background-attachment: fixed; /* Fixes the image in place when scrolling */
        height: 100vh; /* Sets the height to 100% of the viewport height */
        width: 100vw; /* Sets the width to 100% of the viewport width */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #60191900; /* Replace with your desired color */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    
    """,
    unsafe_allow_html=True
)






# Sidebar for user inputs
with st.sidebar:
    st.image(
        "https://stimg2.cardekho.com/images/carNewsimages/userimages/650X420/30183/1672738680556/GeneralNew.jpg",
        width=400
    )
    st.title(":red[Used Car Price Prediction]")
    st.title(":red[Features]")
    # Add logo and title in main area
    m_transmission = st.selectbox(label="Transmission", options=car_df['transmission'].unique())
    m_oem = st.selectbox(label="Car Brand", options=car_df['oem'].unique())
    m_km = st.selectbox(label="Select KMs Driven", options=sorted(car_df['km'].unique().astype(int)))
    m_gear = st.selectbox(label="Number of Gears", options=sorted(car_df['Gear_Box'].unique().astype(int)))
    m_fuel_type = st.selectbox(label="Fuel Type", options=car_df['fueltype'].unique())
    m_bodytype = st.selectbox(label="Body Type", options=car_df['bodytype'].unique())
    m_mileage = st.selectbox(label="Mileage", options=sorted(car_df['Mileage'].unique().astype(float)))
    m_drivetype = st.selectbox(label="Drive Type", options=car_df['Drive_Type'].unique())

    m_modelYear = st.selectbox(label="Model Year", options=sorted(car_df['modelYear'].unique().astype(int)))
    
    m_seats = st.selectbox(label="Number of Seats", options=sorted(car_df['seats'].unique().astype(int)))
    m_ownerNo = st.selectbox(label="Number of Owners", options=sorted(car_df['owner'].unique().astype(int)))
    m_Engine = st.selectbox(label="Engine Displacement", options=sorted(car_df['Engine_CC'].unique().astype(int)))
    
    m_Insurance = st.selectbox(label="Insurance", options=car_df['Insurance_Validity'].unique())
    m_city = st.selectbox(label="City", options=car_df['City'].unique())

    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
        """
    ):
        pred_price_button = st.button("Estimate Used Car Price")
        
if pred_price_button:
    prediction_value = predict_resale_price(m_bodytype, m_seats, m_km, m_modelYear, m_ownerNo, m_Engine, 
                                            m_gear, m_mileage, m_fuel_type, m_transmission, m_Insurance, 
                                            m_oem, m_drivetype, m_city)
    st.subheader(f"The estimated used car price is :blue[₹ {float(prediction_value) / 100000:,.2f} Lakhs]")

st.title('Car Price Distribution by Body Type')

# Bar plot using plotly express
fig = px.scatter(car_df, x='km', y='price', color='fueltype')
st.plotly_chart(fig)


# Scatter plot using plotly
st.title("Kms vs Price Scatter Plot")
# Create the scatter plot
fig = px.scatter(car_df, x='km', y='price', color='fueltype')

# Display the plot with a unique key
st.plotly_chart(fig, key="km_price_scatter")


st.title('Mileage Distribution')



# Histogram using plotly express
fig = px.histogram(car_df, x='Mileage', nbins=30, title='Distribution of Car Mileage')
st.plotly_chart(fig)


st.title('Correlation Heatmap')

# Compute correlation matrix
corr_matrix = car_df[['km','Engine_CC','price','seats','owner']].corr()

# Heatmap using plotly express
fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap')
st.plotly_chart(fig)
