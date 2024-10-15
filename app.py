from flask import Flask, request, jsonify
import numpy as np

app = Flask(_name_)

# Define a route for handling requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data
    data = request.get_json()

    # Use the data to make a prediction
    prediction = np.random.rand(1)[0]

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if _name_ == '_main_':
    app.run(debug=True)