import gradio as gr
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load the scaler and models
scaler = joblib.load("/home/abhirama/Learning-ML/Learning_L_Regression/scaler.pkl")
model_h5 = load_model("/home/abhirama/Learning-ML/Learning_L_Regression/model.h5")
svm = joblib.load("/home/abhirama/Learning-ML/Learning_L_Regression/svm_model.pkl")

def scaler_data(dataframe):

    scaler = StandardScaler()
    X  = scaler.fit_transform(dataframe)   #Features have varying ranges of data and needs to be scaled for prediction

    return X

def predict_particle_from_csv(file, model_choice):
        # Read the CSV file
        data = pd.read_csv(file.name)
        
        # Validate that the required columns are present
        required_columns = [
            "fLength", "fWidth", "fSize", "fConc", "fConc1", 
            "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"
        ]
        if not all(column in data.columns for column in required_columns):
            return "Error: CSV file is missing required columns. Please include all feature columns."
        
        # Scale the input data
        input_scaled = scaler_data(data)
        
        # Select model and make predictions
        if model_choice == "Deep Learning Model":
            # Deep learning model outputs probabilities
            predictions = model_h5.predict(input_scaled)
            labels = ["Gamma" if pred[0] > 0.5 else "Hadron" for pred in predictions]
            
        elif model_choice == "SVM Model":
            predictions = svm.predict(input_scaled)
            labels = ["Gamma" if pred == 1 else "Hadron" for pred in predictions]
        else:
            return "Error: Invalid model choice."
        
        output = pd.DataFrame({
            "Row": range(1, len(labels) + 1),
            "Prediction": labels
        })
        
        return output
    
    
    

interface = gr.Interface(
    fn=predict_particle_from_csv,
    inputs=[
        gr.File(label="Upload CSV File", file_types=[".csv"]),
        gr.Radio(["Deep Learning Model", "SVM Model"], label="Select Model")
    ],
    outputs=gr.Dataframe(headers=["Row", "Prediction"], label="Predictions"),
    title="Particle Classification",
    description="Upload a CSV file containing particle features to predict whether each particle is a Gamma or Hadron. \n\nEnsure your CSV file contains the following columns: \n`fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`.",
    theme="huggingface"
)

# Launch the app
interface.launch()
    