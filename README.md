## House Recommendation Backend
This project is a backend service for a house recommendation system. It uses a machine learning model to predict and recommend houses based on user input. The backend is built using Flask and TensorFlow.

## Features
- **Predict**: Provides house recommendations based on user input such as price, property size, number of beds, number of baths, and state.
- **Test**: A route to check if the backend is functioning correctly.
- **Get States**: Fetches a list of states from a JSON file for use in the application.

## Technologies Used
- **Flask**: A web framework for Python to build the API.
- **TensorFlow**: For loading and using the pre-trained model.
- **Pandas**: For data manipulation and preparation.
- **NumPy**: For numerical operations.
- **Scikit-Learn**: For normalization and label encoding.
- **Flask-CORS**: To handle Cross-Origin Resource Sharing (CORS).

## Setup
## Prerequisites
- Python 3.x
- Flask
- TensorFlow
- Pandas
- NumPy
- Scikit-Learn
- Flask-CORS

## API Endpoints
### GET /test
- **Description:** Check if the backend is working.
- **Response:**
  ```json
  {
    "message": "Backend is working"
  }

## POST /predict
- **Description**: Predict house recommendations based on user input.
- **Request Body (JSON):**
  ```json
  {
    "PRICE": 100000,
    "PROPERTYSQFT": 7000,
    "BEDS": 7,
    "BATH": 8,
    "STATE": "Manhattan"
  }
- **Response**: A list of recommended houses with details including address, price, beds, baths, match score, and coordinates.

## GET /states
- **Description:** Fetch a list of states.
- **Response:** A JSON list of state names

