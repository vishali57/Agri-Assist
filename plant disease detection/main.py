import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image   
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Agri-Assist",
    page_icon="🌿",
    layout="wide"
)

# Load the model only once (to avoid reloading on every prediction)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_plant_disease_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# TensorFlow Model Prediction for Plant Disease
def model_prediction(image):
    try:
        if model is None:
            return None  # Model not loaded

        # Resize and preprocess the image
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
        
        # Predict using the model
        predictions = model.predict(input_arr)

        # Debugging output
        print("Predictions:", predictions)

        # Ensure predictions are valid before calling np.argmax()
        if predictions is not None and len(predictions) > 0:
            result_index = np.argmax(predictions)
            print("Predicted index:", result_index)
            return result_index
        else:
            return None
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return None

# Load the pre-trained model and preprocessor
try:
    dtr = pickle.load(open('dtr.pkl', 'rb'))
    preprocesser = pickle.load(open('preprocessor.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")

# Load dataset for Crop Yield Prediction
try:
    df = pd.read_csv('yield_df.csv')  
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Dummy function for Crop Yield Prediction
def prediction(year, avg_rainfall, pesticides, avg_temp, area, item):
    try:
        features = pd.DataFrame([[year, avg_rainfall, pesticides, avg_temp, area, item]],
                                columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])

        transformed_features = preprocesser.transform(features)
        predicted_yield = dtr.predict(transformed_features)

        return predicted_yield[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a feature", ["🌿 Plant Disease Recognition", "🚜 Crop Yield Prediction"])

st.title("🌾 Agri-Assist Dashboard")

# Plant Disease Recognition Page
if page == "🌿 Plant Disease Recognition":
    st.header("🌿 Plant Disease Recognition")
    st.write("Upload an image of a plant leaf to get a prediction about its health.")

    col1, col2 = st.columns([1, 1])

    with col1:
        test_image = st.file_uploader("Upload a plant image:", type=["jpg", "png", "jpeg"])

        # Hide the file name and uploaded file list after upload (aggressive selector for all Streamlit versions)
        st.markdown("""
            <style>
            /* Hide the file name and uploaded file list */
            [data-testid="stFileUploader"] div[data-testid="stFileUploaderFileName"] {display: none !important;}
            [data-testid="stFileUploader"] ul {display: none !important;}
            [data-testid="stFileUploader"] .uploadedFileName {display: none !important;}
            [data-testid="stFileUploader"] label span {display: none !important;}
            </style>
        """, unsafe_allow_html=True)

        if test_image is not None:
            # Prediction button
            if st.button("🔍 Predict Disease"):
                with st.spinner('Analyzing...'):
                    img_for_pred = Image.open(test_image)
                    result_index = model_prediction(img_for_pred)

                    # List of plant disease classes
                    class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                                  'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
                                  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                                  'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                                  'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                                  'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                                  'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                                  'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                                  'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                                  'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                                  'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                                  'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                                  'Tomato___healthy']

                    # Ensure the predicted index is within bounds
                    if result_index is not None and 0 <= result_index < len(class_name):
                        st.success(f"### 🌱 Prediction: **{class_name[result_index]}**")
                    else:
                        st.error("🚨 Invalid prediction index. Please check the model output.")

    with col2:
        if test_image is not None:
            img_display = Image.open(test_image)
            st.image(img_display, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Please upload an image to see a preview.")


# Crop Yield Prediction Page
elif page == "🚜 Crop Yield Prediction":
    st.header("🚜 Crop Yield Prediction")
    st.write("Fill in the details below to predict the crop yield.")

    if 'Area' in df.columns and 'Item' in df.columns:
        with st.form("yield_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input("Year", min_value=1990, max_value=2030, value=2022)
                avg_rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1485.0, step=10.0)
                area = st.selectbox("Area", options=sorted(df['Area'].unique()))

            with col2:
                pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=121.0, step=5.0)
                avg_temp = st.number_input("Average Temperature (°C)", min_value=-20.0, max_value=60.0, value=16.37, step=0.5)
                item = st.selectbox("Item", options=sorted(df['Item'].unique()))

            submitted = st.form_submit_button("🌾 Predict Yield")

            if submitted:
                with st.spinner('Calculating...'):
                    result = prediction(year, avg_rainfall, pesticides, avg_temp, area, item)

                    if result is not None:
                        st.success(f"### 🌾 Predicted Yield: **{result:.2f} hg/ha**")
                        st.balloons()
                    else:
                        st.error("🚨 Error: Unable to calculate yield.")
    else:
        st.error("🚨 Missing required columns in the dataset. Check 'Area' and 'Item'.")
