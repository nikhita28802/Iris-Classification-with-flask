from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create Flask App
app = Flask(__name__)

# Ensure 'static' directory exists to store images
if not os.path.exists('static'):
    os.mkdir('static')

# Load the Iris Dataset for visualizations
df = pd.read_csv('data/Iris.csv')

# Remove 'Species' column from the dataset when performing visualizations or model predictions
# Assuming the CSV has columns: 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Load the trained model and scaler for predictions
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

# Home page with form to input features for prediction and visualizations
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    pairplot_path = None
    heatmap_path = None

    if request.method == 'POST':
        try:
            # Get form data for prediction
            features = [float(x) for x in request.form.values()]
            features = np.array(features).reshape(1, -1)

            # Scale the input features
            features = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features)[0]
        except ValueError:
            prediction = "Invalid input. Please enter valid numerical values."

        # Generate Pairplot and save as image (excluding 'Species' for features)
        sns.pairplot(df, hue='Species', diag_kind='kde', vars=feature_columns)
        pairplot_path = 'static/pairplot.png'
        plt.savefig(pairplot_path)
        plt.clf()  # Clear the current figure to generate the next plot

        # Generate Correlation Heatmap and save as image
        plt.figure(figsize=(8, 6))
        correlation_matrix = df[feature_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        heatmap_path = 'static/correlation_heatmap.png'
        plt.savefig(heatmap_path)
        plt.clf()

    # Render the template and pass the prediction and image paths
    return render_template('index.html',
                           prediction=prediction,
                           pairplot_path=pairplot_path,
                           heatmap_path=heatmap_path)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
