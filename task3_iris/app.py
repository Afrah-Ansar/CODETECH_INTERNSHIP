from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Class names based on target labels in Iris dataset
class_names = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__)

@app.route('/')
def home():
    return "🌼 Iris Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate request data
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in JSON body"}), 400

    try:
        # Convert input to array and predict
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)[0]
        class_name = class_names[prediction]

        return jsonify({
            "predicted_class": int(prediction),
            "class_name": class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5001)
