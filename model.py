import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


data = pd.read_csv('data/Iris.csv')

# Prepare the data
X = data.drop(['Species'], axis=1)
y = data['Species']

# Split the data (you can skip this if you're not evaluating the model here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Define the model
def load_model():
    # You can load a pre-trained model from a file
    # For simplicity, we're training the model here
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_species(features):
    model = load_model()
    model.fit(features['train'], features['target'])  # Add your training data here
    prediction = model.predict([features['input']])
    return prediction[0]

