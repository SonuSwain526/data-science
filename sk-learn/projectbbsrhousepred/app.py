from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# Define the feature names (order must match training)
num_features = ["longitude","latitude","housing_median_age","total_rooms",
                "total_bedrooms","population","households","median_income"]
cat_features = ["ocean_proximity"]  # categorical feature

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect numeric inputs
        num_values = [float(request.form[feat]) for feat in num_features]
        
        # Collect categorical input
        cat_value = request.form["ocean_proximity"]
        
        # Combine numeric + categorical
        df = pd.DataFrame([num_values + [cat_value]], columns=num_features + cat_features)
        
        # Transform features using pipeline
        transformed_data = pipeline.transform(df)
        
        # Make prediction
        prediction = model.predict(transformed_data)[0]
        return render_template('index.html', prediction_text=f"Predicted House Price: ${prediction:,.2f}")
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
