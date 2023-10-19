from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
with open('model.pkl','rb') as f:
    model = pickle.load(f)
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

@app.route('/predict', methods=['GET','POST'])
def predict():
    request_data = request.get_json() #request.data.decode('ascii')
    request_vals = list(request_data.values())
    scaled_vals = scaler.transform([request_vals])
    prediction = model.predict(scaled_vals)
    return jsonify(
        {
            'result': prediction[0]
        }
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)