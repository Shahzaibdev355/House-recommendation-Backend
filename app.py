import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU


import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import the CORS library
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle



import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

CORS(app)

# Load your model
model = load_model('new_model.h5')

# Load your data
df = pd.read_csv(r"https://raw.githubusercontent.com/Shahzaibdev355/House-recommendation-Backend/master/NY-House-Dataset.csv")

# Initialize the MinMaxScaler and fit it using the training data (same as during training)
ms = MinMaxScaler()
ms.fit(df[['PRICE', 'PROPERTYSQFT', 'BEDS', 'BATH']])

# Load the LabelEncoder for states
with open('state_label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


# Route to test if the backend is working
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Backend is working"}), 200   




@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        try:
            price = float(data.get('PRICE'))
            beds = int(data.get('BEDS'))
            baths = int(data.get('BATH'))
            sqrt = float(data.get('PROPERTYSQFT'))
            state = data.get('STATE').strip()  # State as a string
        except (ValueError, TypeError, AttributeError):
            return jsonify({'error': 'Invalid data type in request'}), 400

        # Check if all required fields are provided
        if None in [price, beds, baths, sqrt, state]:
            return jsonify({'error': 'Missing data in request'}), 400

        # Prepare the user input as a DataFrame (same structure as your training data)
        user_input = pd.DataFrame({
            'PRICE': [price],
            'PROPERTYSQFT': [sqrt],
            'BEDS': [beds],
            'BATH': [baths]
        })

        # Normalize user input with the same MinMaxScaler
        user_input_normalized = ms.transform(user_input)

        # Extract the normalized values
        user_input_price = user_input_normalized[0][0]
        user_input_sqrt = user_input_normalized[0][1]
        user_input_beds = user_input_normalized[0][2]
        user_input_baths = user_input_normalized[0][3]

        # Calculate similarity score based on the difference between the user input and each row in the dataset
        def calculate_similarity(row):
            price_diff = abs(row['PRICE'] - user_input_price) / max(df['PRICE'])
            sqrt_diff = abs(row['PROPERTYSQFT'] - user_input_sqrt) / max(df['PROPERTYSQFT'])
            beds_diff = abs(row['BEDS'] - user_input_beds) / max(df['BEDS'])
            baths_diff = abs(row['BATH'] - user_input_baths) / max(df['BATH'])
            state_diff = 0 if row['STATE'].strip().lower() == state.lower() else 1  # Penalize if state doesn't match

            # Similarity score: lower is better
            return price_diff + sqrt_diff + beds_diff + baths_diff + state_diff

        # Apply the similarity score function to the dataframe
        df['similarity_score'] = df.apply(calculate_similarity, axis=1)

        # Get top 150 recommendations based on the lowest similarity score
        top_recommendations = df.sort_values(by='similarity_score').head(150)

        # Format recommendations for the response
        results = []
        for i, row in top_recommendations.iterrows():
            results.append({
                'Address': row['FORMATTED_ADDRESS'],
                'Price': row['PRICE'],
                'Beds': row['BEDS'],
                'Baths': row['BATH'],
                'PropertySqrt': f"{round(row['PROPERTYSQFT'])} sq ft",
                'State': row['STATE'],
                'Latitude': row['LATITUDE'],
                'Longitude': row['LONGITUDE']
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    




@app.route('/states', methods=['GET'])
def get_states():
    try:
        # Fetch the JSON data from the URL
        response = requests.get('https://raw.githubusercontent.com/Shahzaibdev355/House-recommendation-Backend/master/state_name.json')
        
        # Check if the request was successful
        if response.status_code == 200:
            # Return the JSON content
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to fetch state data'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    app.run(debug=False)
