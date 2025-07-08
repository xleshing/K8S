from flask import Flask, request, jsonify
from time_aware_lstm_model import get_pod_data, prepare_data, train_model, predict_at_time

app = Flask(__name__)

@app.route('/predict_time', methods=['GET'])
def predict_time():
    pod = request.args.get('pod')
    target_time = request.args.get('time')  # 例：2025-05-28 12:00

    if not pod or '/' not in pod:
        return jsonify({'error': 'Invalid pod format. Use namespace/pod_name'}), 400
    if not target_time:
        return jsonify({'error': 'Missing time query parameter (e.g. 2025-05-28 12:00)'}), 400

    namespace, pod_name = pod.split('/', 1)
    df = get_pod_data(namespace, pod_name)
    if len(df) < 31:
        return jsonify({'error': 'Not enough data to make prediction'}), 400

    X, y, scaler = prepare_data(df)
    model = train_model(X, y)
    prediction = predict_at_time(model, df, target_time, scaler)
    return jsonify({'pod': pod, 'predicted_cpu_usage_at': target_time, 'cpu': prediction})

@app.route('/history', methods=['GET'])
def history():
    pod = request.args.get('pod')
    if not pod or '/' not in pod:
        return jsonify({'error': 'Invalid pod format. Use namespace/pod_name'}), 400
    namespace, pod_name = pod.split('/', 1)
    df = get_pod_data(namespace, pod_name)
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)