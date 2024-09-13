from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import the CORS library
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

# Enable CORS for all routes and allow requests from any origin
CORS(app)

# Load your model
model = load_model('new_model.h5')

# Load your data
df = pd.read_csv("https://raw.githubusercontent.com/Shahzaibdev355/House-recommendation-Backend/master/NY-House-Dataset.csv")

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
        price = data.get('PRICE')
        beds = data.get('BEDS')
        baths = data.get('BATH')
        state = data.get('STATE')  # User input state as a string
        sqrt = data.get('PROPERTYSQFT')

        # Check if all required fields are provided
        if None in [price, beds, baths, state, sqrt]:
            return jsonify({'error': 'Missing data in request'}), 400

        # Transform the state input using the LabelEncoder
        try:
            encoded_state = le.transform([state])[0]
        except ValueError:
            return jsonify({'error': 'Invalid state value'}), 400

        # Prepare the user input as a DataFrame (same structure as your training data)


        user_input = pd.DataFrame({
            'PRICE': [price],
            'PROPERTYSQFT': [sqrt],
            'BEDS': [beds],
            'BATH': [baths]
        })

        # Normalize user input with same MinMaxScaler
        user_input_normalized = ms.transform(user_input)





        # Extract the normalized values
        user_input_price = np.array([user_input_normalized[0][0]] * len(df))
        user_input_sqrt = np.array([user_input_normalized[0][1]] * len(df))
        user_input_beds = np.array([user_input_normalized[0][2]] * len(df))
        user_input_baths = np.array([user_input_normalized[0][3]] * len(df))

        # Handle the encoded state input (assuming it's categorical)
        user_input_state = np.array([encoded_state] * len(df))

        # Make predictions
        predictions = model.predict([user_input_price, user_input_state, user_input_beds, user_input_baths])
        df['match_score'] = predictions

        # Get top 10 recommendations
        top_10_recommendations = df.sort_values(by='match_score', ascending=False).head(40)

        # Format recommendations for response
        results = []
        for i, row in top_10_recommendations.iterrows():
            results.append({
                'Address': row['FORMATTED_ADDRESS'],
                'Price': row['PRICE'],
                'Beds': row['BEDS'],
                'Baths': row['BATH'],
                'Match Score': float(row['match_score']),  # Convert to float for JSON serialization
                'PropertySqrt': f" {round(row['PROPERTYSQFT'])} sq",

                'Longitute': row['LATITUDE'],
                'Latitude': row['LONGITUDE']
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    




@app.route('/states', methods=['GET'])
def get_states():
    return send_from_directory(directory='.', path='https://raw.githubusercontent.com/Shahzaibdev355/House-recommendation-Backend/master/state_name.json', as_attachment=False)






if __name__ == '__main__':
    app.run(debug=True)
