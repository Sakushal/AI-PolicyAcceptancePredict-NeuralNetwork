from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Load the trained TensorFlow model
model = tf.keras.models.load_model("policy_acceptance_tf_nn_model_new.h5")

# Load the preprocessor
preprocessor = joblib.load("preprocessor_new.pkl")

# Create a Flask app
app = Flask(__name__)

# Define the API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the data into a DataFrame
        new_proposal = pd.DataFrame({
        "age": [int(data["age"])],  # Convert age to int
        "income": [int(data["income"])],  # Convert income to int
        "height_m": [float(data["height_m"])],  # Convert height to float
        "weight_kg": [float(data["weight_kg"])],  # Convert weight to float
        "bmi": [float(data["weight_kg"]) / (float(data["height_m"]) ** 2)],  # Convert & calculate BMI
        "health_history": [data["health_history"]],
        "marital_status": [data["marital_status"]],
        "family_history": [data["family_history"]],
        "smoker": [data["smoker"]],
        "alcohol_consumption": [data["alcohol_consumption"]],
        "occupation": [data["occupation"]]
        })

        # Preprocess the input data
        new_proposal_processed = preprocessor.transform(new_proposal)

        # Make a prediction using the neural network
        acceptance_probability = model.predict(new_proposal_processed)[0, 0] * 100
        prediction = int(acceptance_probability > 50)  # Threshold at 50%

        # Prepare the decision message
        if prediction == 1:
            decision = f"Prediction: There is a ({acceptance_probability:.0f}%) chance that this proposal might get Accepted."
        else:
            decision = f"Prediction: There is a ({100 - acceptance_probability:.0f}%) chance that this proposal might get Rejected."

        # Return the prediction as a JSON response
        return jsonify({
            "decision": decision
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
