from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load Dataset
data = pd.read_csv('D:/Coding/Machine Learning/Datasets/synthetic_cognitive_load_large_data.csv')

# Preprocessing
X = data[['Heart Rate', 'Skin Conductance', 'Eye Blink Rate', 'Pupil Dilation']]
y = data['Cognitive Load']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(clf, 'cognitive_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ""
    
    if request.method == 'POST':
        heart_rate = float(request.form['heart_rate'])
        skin_conductance = float(request.form['skin_conductance'])
        eye_blink_rate = float(request.form['eye_blink_rate'])
        pupil_dilation = float(request.form['pupil_dilation'])
        
        # Load the model and scaler
        clf = joblib.load('cognitive_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Prepare input for prediction
        input_data = [[heart_rate, skin_conductance, eye_blink_rate, pupil_dilation]]
        input_scaled = scaler.transform(input_data)

        # Make the prediction
        predicted_load = clf.predict(input_scaled)
        prediction = predicted_load[0]  # Get the predicted class
        
        # Set the message based on the prediction
        if prediction == 0:
            message = "Congratulations! You are under no load."
        elif prediction == 1:
            message = "Be patient and calm."
        elif prediction == 2:
            message = "You need to be calm immediately. You are in the brain damage zone."

    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
