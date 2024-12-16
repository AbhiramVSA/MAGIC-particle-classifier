import gradio as gr
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.svm import SVC

# Load the scaler and models
scaler = joblib.load("/home/abhirama/Learning-ML/Learning_L_Regression/scaler.pkl")
model_h5 = load_model("/home/abhirama/Learning-ML/Learning_L_Regression/model.h5")

# Attempt to load SVM model with error handling
try:
    svm_model = joblib.load("Learning_L_Regression/svm_model.pkl")
    
    # Check if the model supports probability predictions
    if not hasattr(svm_model, 'predict_proba'):
        # Recreate the model with probability support if needed
        svm_model = SVC(
            kernel=svm_model.kernel, 
            C=svm_model.C, 
            probability=True,  # Ensure probability predictions are enabled
            random_state=42
        )
        # You might need to retrain the model here if the original model can't be modified
        print("Warning: Recreated SVM model with probability support")

except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None

# Debug function to investigate SVM model behavior
def debug_svm_model(file):
    """
    Diagnostics function to understand SVM model prediction behavior
    """
    try:
        # Check if SVM model is loaded
        if svm_model is None:
            return "Error: SVM model is not properly loaded."
        
        # Read the CSV file
        data = pd.read_csv(file.name)
        
        # Validate required columns
        required_columns = [
            "fLength", "fWidth", "fSize", "fConc", "fConc1", 
            "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"
        ]
        if not all(column in data.columns for column in required_columns):
            return "Error: CSV file is missing required columns."
        
        # Scale the input data
        input_scaled = scaler.transform(data[required_columns])
        
        # Investigate SVM model predictions
        print("SVM Model Information:")
        try:
            print("Model Classes:", svm_model.classes_)
        except Exception as e:
            print(f"Could not retrieve model classes: {e}")
        
        # Detailed prediction investigation
        try:
            # Attempt to get raw and probability predictions
            raw_predictions = svm_model.predict(input_scaled)
            
            # Handle probability predictions differently
            try:
                proba_predictions = svm_model.predict_proba(input_scaled)
                print("\nProbability Predictions Available")
            except Exception as e:
                print(f"\nProbability predictions not available: {e}")
                proba_predictions = None
            
            print("\nRaw Predictions:")
            print(raw_predictions)
            
            # Prepare detailed output
            output_data = {
                "Row": range(1, len(raw_predictions) + 1),
                "Raw Prediction": raw_predictions
            }
            
            # Add probability columns if available
            if proba_predictions is not None:
                if proba_predictions.shape[1] == 2:
                    output_data["Probability Gamma"] = proba_predictions[:, 1]
                    output_data["Probability Hadron"] = proba_predictions[:, 0]
                else:
                    output_data["Probabilities"] = list(proba_predictions)
            
            output = pd.DataFrame(output_data)
            return output
        
        except Exception as e:
            return f"Error in SVM prediction: {str(e)}"
    
    except Exception as e:
        return f"Error in debugging: {str(e)}"

# Updated prediction function with robust error handling
def predict_particle_from_csv(file, model_choice):
    try:
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
        input_scaled = scaler.transform(data[required_columns])
        
        # Select model and make predictions
        if model_choice == "Deep Learning Model":
            # Deep learning model outputs probabilities
            predictions = model_h5.predict(input_scaled)
            labels = ["Gamma" if pred[0] > 0.5 else "Hadron" for pred in predictions]
        
        elif model_choice == "SVM Model":
            # Check if SVM model is loaded
            if svm_model is None:
                return "Error: SVM model is not properly loaded."
            
            # Improved SVM prediction handling
            try:
                # Try probability predictions first
                try:
                    proba_predictions = svm_model.predict_proba(input_scaled)
                    labels = ["Gamma" if pred[1] > 0.5 else "Hadron" for pred in proba_predictions]
                except Exception:
                    # Fallback to standard prediction if probability prediction fails
                    raw_predictions = svm_model.predict(input_scaled)
                    labels = ["Gamma" if pred == 1 else "Hadron" for pred in raw_predictions]
            
            except Exception as e:
                return f"Error with SVM prediction: {str(e)}"
        
        else:
            return "Error: Invalid model choice."
        
        # Prepare the output DataFrame
        output = pd.DataFrame({
            "Row": range(1, len(labels) + 1),
            "Prediction": labels
        })
        
        return output
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create the Gradio interface with added debug option
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

# Optional: Debug Interface to investigate model behavior
debug_interface = gr.Interface(
    fn=debug_svm_model,
    inputs=gr.File(label="Upload CSV File for Debugging", file_types=[".csv"]),
    outputs=gr.Dataframe(label="SVM Model Debug Output"),
    title="SVM Model Debug",
    description="Upload a CSV to investigate SVM model prediction details."
)

# Combine interfaces
combined_interface = gr.TabbedInterface(
    [interface, debug_interface], 
    ["Prediction", "Debug SVM"]
)

# Launch the app
if __name__ == "__main__":
    combined_interface.launch(share=True)