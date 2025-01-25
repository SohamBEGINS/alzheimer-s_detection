from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import numpy as np
from keras.preprocessing import image
from keras.applications import DenseNet201
import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import time

app = Flask(__name__)

# Load models and scaler
scaler = joblib.load('scaler.pkl')  # Load the scaler
stacking_model = joblib.load('stacking_model.pkl')  # Load the stacking model

model_dir = r'C:\Users\soham\OneDrive\Desktop\src\ct_scan_models'


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

from flask import Flask, render_template, request, redirect, send_file
import os
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors



@app.route('/medical_info', methods=['GET', 'POST'])
def get_medical_info():
    if request.method == 'POST':
        try:
            input_features = []

            def convert_to_mg_dl(value, unit):
                if unit == "mmol/L":
                    return round(value * 38.67, 2)
                return round(value, 2)

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

            functional_assessment = request.form['FunctionalAssessment']
            if functional_assessment == "Not Available":
                return redirect("https://www.compassus.com/healthcare-professionals/determining-eligibility/functional-assessment-staging-tool-fast-scale-for-dementia/")
            else:
                input_features.append(round(float(functional_assessment), 2))

            adl_value = request.form['ADL']
            if adl_value == "Not Available":
                return redirect("https://www.mdcalc.com/calc/3912/barthel-index-activities-daily-living-adl#evidence")
            else:
                input_features.append(round(float(adl_value), 2))

            input_features.extend([
                int(request.form['MemoryComplaints']),
                int(request.form['BehavioralProblems']),
                int(request.form['Confusion']),
                int(request.form['Disorientation']),
                int(request.form['PersonalityChanges']),
                int(request.form['DifficultyCompletingTasks']),
                int(request.form['Forgetfulness'])
            ])

            scaled_features = scaler.transform([input_features])
            prediction = stacking_model.predict(scaled_features)

            diagnosis = "Positive for Alzheimer's" if prediction[0] == 1 else "Negative for Alzheimer's"

            # Generate PDF with user inputs and prediction result
            pdf_filename = f"medical_report_{int(time.time())}.pdf"  # Unique filename using timestamp
            pdf_path = os.path.join(r'C:\Users\soham\OneDrive\Desktop\src\temp_save', pdf_filename)
          

            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

            data = [
              ["Feature", "Value"],
              ["Age", input_features[0]],
              ["Gender", input_features[1]],
              ["Ethnicity", input_features[2]],
              ["Education Level", input_features[3]],
              ["BMI", input_features[4]],
              ["Smoking", input_features[5]],
              ["Family History of Alzheimer's", input_features[6]],
              ["Cardiovascular Disease", input_features[7]],
              ["Diabetes", input_features[8]],
              ["Depression", input_features[9]],
              ["Head Injury", input_features[10]],
              ["Hypertension", input_features[11]],
              ["Systolic BP", input_features[12]],
              ["Diastolic BP", input_features[13]],
              ["Cholesterol Total", input_features[14]],
              ["Cholesterol LDL", input_features[15]],
              ["Cholesterol HDL", input_features[16]],
              ["Cholesterol Triglycerides", input_features[17]],
              ["MMSE", input_features[18]],
              ["Functional Assessment", input_features[19]],
              ["Memory Complaints", input_features[21]],  # New parameters start here
              ["Behavioral Problems", input_features[22]],
              ["ADL Value", input_features[20]],  # ADL placed after Behavioral Problems
              ["Confusion", input_features[23]],
              ["Disorientation", input_features[24]],
              ["Personality Changes", input_features[25]],
              ["Difficulty Completing Tasks", input_features[26]],
              ["Forgetfulness", input_features[27]],
              ["Diagnosis", diagnosis]  # Diagnosis as the last parameter
                                               ]

            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            elements = [table]
            doc.build(elements)

            with open(pdf_path, 'wb') as f:
                f.write(pdf_buffer.getvalue())

            return render_template('result.html', diagnosis=diagnosis, pdf_filename=pdf_filename)

        except Exception as e:
            return f"Error: {str(e)}"

    else:
        return render_template('predict_medical.html')

@app.route('/trigger_pdf_generation/<filename>', methods=['GET'])
def trigger_pdf_generation(filename):
    try:
        # Construct the full path to the PDF file
        pdf_path = os.path.join(r'C:\Users\soham\OneDrive\Desktop\src\temp_save', filename)
        print(f"PDF Path: {pdf_path}")  # Debugging line
       
        if os.path.exists(pdf_path):
            return send_file(pdf_path, as_attachment=False, mimetype='application/pdf')
        else:
            return "PDF file does not exist.", 404
    except Exception as e:
        return f"Error: {str(e)}", 500





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
            return render_template('ct_scan_result.html', diagnosis=class_label[predicted_class[0]])

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)