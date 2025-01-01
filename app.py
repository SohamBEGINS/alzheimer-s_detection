from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet201
import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

# Load models and scaler
scaler = joblib.load('scaler.pkl')  # Load the scaler
stacking_model = joblib.load('stacking_model.pkl')  # Load the stacking model

model_dir = r'C:\Users\soham\OneDrive\Desktop\alzheimers_website\ct_scan_models'

# Load DenseNet model weights
densenet_model = DenseNet201(weights=None, include_top=False, input_shape=(224, 224, 3))
densenet_model.load_weights(os.path.join(model_dir, 'densenet_weights.h5'))  # Load DenseNet model weights

# Rebuild the ANN model architecture and load weights
ann_model = Sequential([
    Dense(512, input_dim=2000, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # Output layer (softmax for binary classification)
])
ann_model.load_weights(os.path.join(model_dir, 'ann_weights.h5'))  # Load ANN model weights

# Load PCA model
pca = joblib.load(os.path.join(model_dir, 'pca_model.pkl'))  # Load PCA model

# Sample credentials for demonstration purposes
USER_CREDENTIALS = {"username": "admin", "password": "password123"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if credentials are correct
        if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
            return redirect(url_for('dashboard'))  # Redirect to dashboard or main page after login
        else:
            error = "Invalid username or password. Please try again."
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/medical_info', methods=['GET', 'POST'])
def get_medical_info():
    if request.method == 'POST':
        try:
            # Get data from form
            input_features = []

            def convert_to_mg_dl(value, unit):
                if unit == "mmol/L":
                    return round(value * 38.67, 2)
                return round(value, 2)

            # Collecting input features from the form
            input_features.append(int(request.form['Age']))
            input_features.append(int(request.form['Gender']))
            input_features.append(int(request.form['Ethnicity']))
            input_features.append(int(request.form['EducationLevel']))
            bmi_value = request.form['BMI']
            if bmi_value == "Not Available":
                return redirect("https://www.calculator.net/bmi-calculator.html")
            else:
                input_features.append(round(float(bmi_value), 2))
            input_features.append(int(request.form['Smoking']))
            input_features.append(int(request.form['FamilyHistoryAlzheimers']))
            input_features.append(int(request.form['CardiovascularDisease']))
            input_features.append(int(request.form['Diabetes']))
            input_features.append(int(request.form['Depression']))
            input_features.append(int(request.form['HeadInjury']))
            input_features.append(int(request.form['Hypertension']))
            input_features.append(int(round(float(request.form['SystolicBP']))))
            input_features.append(int(round(float(request.form['DiastolicBP']))))
            input_features.append(convert_to_mg_dl(float(request.form['CholesterolTotal']), request.form['CholesterolTotalUnit']))
            input_features.append(convert_to_mg_dl(float(request.form['CholesterolLDL']), request.form['CholesterolLDLUnit']))
            input_features.append(convert_to_mg_dl(float(request.form['CholesterolHDL']), request.form['CholesterolHDLUnit']))
            input_features.append(convert_to_mg_dl(float(request.form['CholesterolTriglycerides']), request.form['CholesterolTriglyceridesUnit']))
            mmse_value = request.form['MMSE']
            if mmse_value == "Not Available":
                return redirect("https://compendiumapp.com/post_4xQIen-Ly")
            else:
                input_features.append(float(mmse_value))
            adl_value = request.form['FunctionalAssessment']
            if adl_value == "Not Available":
                return redirect("https://www.mdcalc.com/calc/3912/barthel-index-activities-daily-living-adl#evidence")
            else:
                input_features.append(round(float(adl_value), 2))
            input_features.extend([
                int(request.form['MemoryComplaints']),
                int(request.form['BehavioralProblems']),
                int(round(float(request.form['ADL']))),
                int(request.form['Confusion']),
                int(request.form['Disorientation']),
                int(request.form['PersonalityChanges']),
                int(request.form['DifficultyCompletingTasks']),
                int(request.form['Forgetfulness'])
            ])

            # Preprocess and predict
            scaled_features = scaler.transform([input_features])
            prediction = stacking_model.predict(scaled_features)

            # Output result
            diagnosis = "Positive for Alzheimer's" if prediction[0] == 1 else "Negative for Alzheimer's"
            return render_template('result.html', diagnosis=diagnosis)

        except Exception as e:
            return f"Error: {str(e)}"

    else:
        return render_template('predict_medical.html')

from flask import jsonify
import time

import time

@app.route('/upload_ct_scan', methods=['POST'])
def upload_ct_scan():
    try:
        # Get image file from POST request
        file = request.files['ct_scan']

        # Create a temporary file and save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)  # Save the file temporarily
            tmp.close()  # Close the file so it can be used by load_img

            # Preprocess the image (resize, rescale, etc.)
            img = image.load_img(tmp.name, target_size=(224, 224))  # Pass the temporary file path
            img_array = image.img_to_array(img)  # Convert image to array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Rescale the image

            # # Simulate some processing delay here before predicting
            # time.sleep(5)  # Wait for 5 seconds before starting model prediction (adjust as needed)

            # Extract features using the DenseNet model
            img_features = densenet_model.predict(img_array)  # Use DenseNet for feature extraction
            img_features_flat = img_features.reshape(1, -1)
            img_features_pca = pca.transform(img_features_flat)  # Apply PCA transformation

            # Predict using the trained ANN model
            prediction = ann_model.predict(img_features_pca)
            predicted_class = np.argmax(prediction, axis=1)
            class_label = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

            # Simulate more processing time (you can adjust the time here as needed)
            time.sleep(3)  # Add another delay to simulate time spent on prediction

            # Return prediction result
            return render_template('ct_scan_result.html', diagnosis=f"Predicted: {class_label[predicted_class[0]]}")

    except Exception as e:
        return f"Error: {str(e)}"







if __name__ == '__main__':
    app.run(debug=True)